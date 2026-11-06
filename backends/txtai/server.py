"""txtai's ANN as a vector index for kurag (VECTOR_BACKEND=txtai): one index per namespace, keyed by kurag's chunk
row ids. kurag keeps everything else, including the float16 copy of every vector it fills this index from.

Settings (environment), as txtai-kura used them: faiss, 8-bit scalar quantizer (SQ8); an index built on fewer than
IVF_MIN vectors is exact (IDMap), a larger one is IVF with 4 * sqrt(n) cells of which NPROBE (default: a quarter) are
searched. kurag fills an index in batches ("fill"), then asks to build it once, so the choice sees every vector;
vectors added later go into the index as it is (kurag fills it again on /rebuild).

    POST /ns/<namespace>/count   {}                                  -> {"count": n}
    POST /ns/<namespace>/clear   {}
    POST /ns/<namespace>/add     {"ids": [...], "dims": d, "vectors": base64 of little-endian float32, "fill": false}
    POST /ns/<namespace>/build   {}   builds the index from the vectors sent with "fill": true since the last clear
    POST /ns/<namespace>/remove  {"ids": [...]}
    POST /ns/<namespace>/search  {"vector": base64 float32, "k": n}   -> {"hits": [[id, cosine], ...]}
    GET  /health

Indexes are written to DATA_DIR (default ./data) every SAVE_SECONDS after a change and on SIGTERM.
"""
import base64
import json
import math
import os
import re
import signal
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from txtai.ann import ANNFactory

DATA = Path(os.environ.get("DATA_DIR") or "data")
LISTEN = os.environ.get("LISTEN") or "0.0.0.0:9670"
QUANTIZE = int(os.environ.get("QUANTIZE") or 8)
IVF_MIN = int(os.environ.get("IVF_MIN") or 100_000)
NPROBE = int(os.environ.get("NPROBE") or 0)
SAVE_SECONDS = float(os.environ.get("SAVE_SECONDS") or 60)
NAME = re.compile(r"^[a-z][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)*$")


def settings(dims: int, count: int) -> dict:
    storage = f"SQ{QUANTIZE}" if QUANTIZE else "Flat"
    f = {"quantize": QUANTIZE} if QUANTIZE else {}
    if count < IVF_MIN:
        f["components"] = f"IDMap,{storage}"
    else:
        cells = max(min(round(4 * math.sqrt(count)), count // 39), 1)  # as txtai: at least 39 vectors per cell
        f["components"] = f"IVF{cells},{storage}"
        f["nprobe"] = NPROBE or max(1, cells // 4)
    return {"backend": "faiss", "dimensions": dims, "faiss": f}


class Index:
    """One namespace: txtai's ANN (ids 0, 1, 2 ... in the order added) and kurag's row id of each."""

    def __init__(self, name: str):
        self.dir = DATA / name
        self.ann, self.vids, self.dirty = None, np.zeros(0, dtype=np.int64), False
        self.pending: list[tuple[list[int], np.ndarray]] = []
        meta = self.dir / "meta.json"
        if meta.exists():
            self.ann = ANNFactory.create(json.loads(meta.read_text()))
            self.ann.load(str(self.dir / "ann"))
            self.vids = np.load(self.dir / "vids.npy")
        self.where = {int(v): i for i, v in enumerate(self.vids) if v >= 0}

    def count(self) -> int:
        return len(self.where)

    def clear(self) -> None:
        if self.ann is not None:
            self.ann.close()
        self.ann, self.vids, self.where, self.dirty = None, np.zeros(0, dtype=np.int64), {}, False
        self.pending = []
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self) -> None:
        if self.pending:
            ids = [i for part, _ in self.pending for i in part]
            vectors = np.concatenate([v for _, v in self.pending])
            self.pending = []
            self.add(ids, vectors)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        stale = [i for i in ids if i in self.where]
        if stale:
            self.remove(stale)
        if self.ann is None:
            self.ann = ANNFactory.create(settings(vectors.shape[1], len(vectors)))
            self.ann.index(vectors)
        else:
            self.ann.append(vectors)
        start = len(self.vids)
        self.vids = np.concatenate([self.vids, np.array(ids, dtype=np.int64)])
        self.where.update({v: start + i for i, v in enumerate(ids)})
        self.dirty = True

    def remove(self, ids: list[int]) -> None:
        aids = [self.where.pop(i) for i in ids if i in self.where]
        if aids and self.ann is not None:
            self.ann.delete(aids)
            self.vids[aids] = -1
            self.dirty = True

    def search(self, q: np.ndarray, k: int) -> list[list]:
        if self.ann is None or not self.where:
            return []
        rows = self.ann.search(q.reshape(1, -1), k)[0]
        return [[int(self.vids[a]), float(s)] for a, s in rows if a >= 0 and self.vids[a] >= 0]

    def save(self) -> None:
        if not self.dirty or self.ann is None:
            return
        tmp = self.dir.with_name(self.dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.ann.save(str(tmp / "ann"))
        np.save(tmp / "vids.npy", self.vids)
        (tmp / "meta.json").write_text(json.dumps(self.ann.config))
        shutil.rmtree(self.dir, ignore_errors=True)
        os.replace(tmp, self.dir)
        self.dirty = False


lock = threading.Lock()
indexes: dict[str, Index] = {}


def index(name: str) -> Index:
    if name not in indexes:
        indexes[name] = Index(name)
    return indexes[name]


def f32(b64: str, dims: int = 0) -> np.ndarray:
    v = np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(np.float32)
    return v.reshape(-1, dims) if dims else v


def save_all() -> None:
    with lock:
        for i in indexes.values():
            i.save()


class Handler(BaseHTTPRequestHandler):
    def reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, {"ok": True, "namespaces": len(indexes)})
        self.reply(404, {"detail": "not found"})

    def do_POST(self):
        m = re.fullmatch(r"/ns/([^/]+)/(count|clear|add|build|remove|search)", self.path)
        if not m or not NAME.match(m.group(1)):
            return self.reply(404, {"detail": "not found"})
        name, op = m.groups()
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            with lock:
                i = index(name)
                if op == "count":
                    return self.reply(200, {"count": i.count()})
                if op == "clear":
                    i.clear()
                elif op == "add" and req.get("fill"):
                    i.pending.append(([int(x) for x in req["ids"]], f32(req["vectors"], int(req["dims"]))))
                elif op == "add":
                    i.add([int(x) for x in req["ids"]], f32(req["vectors"], int(req["dims"])))
                elif op == "build":
                    i.build()
                elif op == "remove":
                    i.remove([int(x) for x in req["ids"]])
                else:
                    return self.reply(200, {"hits": i.search(f32(req["vector"]), int(req["k"]))})
            self.reply(200, {})
        except Exception as e:  # noqa: BLE001 - report to kurag, which fills the index again
            self.reply(500, {"detail": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):
        pass


def main() -> None:
    host, port = LISTEN.rsplit(":", 1)
    server = ThreadingHTTPServer((host, int(port)), Handler)

    def saver():
        while True:
            time.sleep(SAVE_SECONDS)
            save_all()

    threading.Thread(target=saver, daemon=True).start()

    def stop(*_):
        save_all()
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"kurag-txtai: listening on {LISTEN}, data in {DATA}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
