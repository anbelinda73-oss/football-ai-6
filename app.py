import os
import uuid
import cv2
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO


app = FastAPI(title="FOOTBALL AI 6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

D = Path(tempfile.gettempdir()) / "football_ai"
D.mkdir(exist_ok=True)

model = YOLO(os.getenv("YOLO_MODEL", "yolo26n.pt"))

# 작업 상태 저장
jobs = {}


@app.get("/")
def root():
    return {
        "service": "FOOTBALL AI 6.1",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "YOLO + BoT-SORT",
        "version": "6.1"
    }


def run_analysis(job, inp, out, target_x, target_y, target_time):
    try:
        jobs[job] = {"status": "processing", "progress": 0}

        cap = cv2.VideoCapture(str(inp))

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not W or not H:
            jobs[job] = {
                "status": "error",
                "message": "영상 파일을 읽을 수 없습니다."
            }
            cap.release()
            return

        # 중요:
        # target_x / target_y는 이미 픽셀 좌표이므로
        # W, H를 다시 곱하지 않습니다.
        px = float(target_x)
        py = float(target_y)

        target_frame = int(max(0, target_time) * fps)

        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )

        selected = None
        last = None
        lost = 0
        i = 0

        while True:
            ok, fr = cap.read()

            if not ok:
                break

            r = model.track(
                fr,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                verbose=False
            )[0]

            cs = []

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().tolist()

                for b, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(float, b)

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    cs.append(
                        (tid, x1, y1, x2, y2, cx, cy)
                    )

            # 지정한 시간 이후 처음 한 번만 선수 선택
            if i >= target_frame and selected is None and cs:

                inside = [
                    c for c in cs
                    if c[1] <= px <= c[3]
                    and c[2] <= py <= c[4]
                ]

                pool = inside or cs

                selected = min(
                    pool,
                    key=lambda c:
                    (c[5] - px) ** 2 +
                    (c[6] - py) ** 2
                )[0]

            cur = next(
                (c for c in cs if c[0] == selected),
                None
            )

            if cur:
                _, x1, y1, x2, y2, cx, cy = cur

                last = (cx, cy)
                lost = 0

                cv2.rectangle(
                    fr,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    3
                )

                cv2.circle(
                    fr,
                    (int(cx), int(cy)),
                    12,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    fr,
                    "26",
                    (int(x1), max(24, int(y1) - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

            elif selected is not None:
                lost += 1

                if last and lost < 30:
                    cv2.circle(
                        fr,
                        (int(last[0]), int(last[1])),
                        12,
                        (0, 255, 255),
                        2
                    )

            writer.write(fr)

            i += 1

            if total > 0:
                progress = int(i / total * 100)
                jobs[job]["progress"] = min(progress, 99)

        cap.release()
        writer.release()

        if selected is None:
            jobs[job] = {
                "status": "error",
                "message":
                "선수 검출 실패: 선수가 잘 보이는 장면에서 몸 가운데를 지정하세요."
            }
            return

        jobs[job] = {
            "status": "done",
            "progress": 100,
            "video_url": f"/result/{job}"
        }

    except Exception as e:
        jobs[job] = {
            "status": "error",
            "message": str(e)
        }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    target_x: float = Form(...),
    target_y: float = Form(...),
    target_time: float = Form(0)
):
    job = uuid.uuid4().hex

    inp = D / f"{job}.mp4"
    out = D / f"{job}_tracked.mp4"

    # 업로드 파일 저장
    with open(inp, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)

            if not chunk:
                break

            f.write(chunk)

    jobs[job] = {
        "status": "queued",
        "progress": 0
    }

    # 분석은 별도 스레드에서 실행
    thread = threading.Thread(
        target=run_analysis,
        args=(
            job,
            inp,
            out,
            target_x,
            target_y,
            target_time
        ),
        daemon=True
    )

    thread.start()

    # 기다리지 않고 즉시 응답
    return {
        "job": job,
        "status": "processing",
        "status_url": f"/status/{job}",
        "result_url": f"/result/{job}"
    }


@app.get("/status/{job}")
def status(job: str):
    if job not in jobs:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")

    return jobs[job]


@app.get("/result/{job}")
def result(job: str):

    if job not in jobs:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")

    state = jobs[job]

    if state["status"] == "error":
        raise HTTPException(
            422,
            state.get("message", "분석 실패")
        )

    if state["status"] != "done":
        return {
            "status": state["status"],
            "progress": state.get("progress", 0),
            "message": "아직 분석 중입니다."
        }

    p = D / f"{job}_tracked.mp4"

    if not p.exists():
        raise HTTPException(404, "결과 영상이 없습니다.")

    return FileResponse(
        p,
        media_type="video/mp4",
        filename="football_ai_26_tracked.mp4"
    )
