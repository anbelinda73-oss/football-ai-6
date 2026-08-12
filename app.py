import os,uuid,cv2,tempfile
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
app=FastAPI(title="FOOTBALL AI 6.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
D=Path(tempfile.gettempdir())/"football_ai";D.mkdir(exist_ok=True)
model=YOLO(os.getenv("YOLO_MODEL","yolo26n.pt"))
@app.get("/health")
def health():return {"ok":True,"engine":"YOLO+BoT-SORT"}
@app.post("/analyze")
async def analyze(video:UploadFile=File(...),target_x:float=Form(...),target_y:float=Form(...),target_time:float=Form(0)):
 j=uuid.uuid4().hex; inp=D/f"{j}.mp4"; out=D/f"{j}_tracked.mp4";inp.write_bytes(await video.read())
 cap=cv2.VideoCapture(str(inp));fps=cap.get(cv2.CAP_PROP_FPS) or 30;W=int(cap.get(3));H=int(cap.get(4))
 if not W or not H:raise HTTPException(400,"영상 파일을 읽을 수 없습니다.")
 px,py=target_x*W,target_y*H;target_frame=int(max(0,target_time)*fps)
 writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
 selected=None;last=None;lost=0;i=0
 while True:
  ok,fr=cap.read()
  if not ok:break
  r=model.track(fr,persist=True,tracker="botsort.yaml",classes=[0],verbose=False)[0]
  cs=[]
  if r.boxes is not None and r.boxes.id is not None:
   for b,tid in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.id.int().cpu().tolist()):
    x1,y1,x2,y2=map(float,b);cs.append((tid,x1,y1,x2,y2,(x1+x2)/2,(y1+y2)/2))
  if i>=target_frame and selected is None and cs:
   inside=[c for c in cs if c[1]<=px<=c[3] and c[2]<=py<=c[4]]
   pool=inside or cs; selected=min(pool,key=lambda c:(c[5]-px)**2+(c[6]-py)**2)[0]
  cur=next((c for c in cs if c[0]==selected),None)
  if cur:
   _,x1,y1,x2,y2,cx,cy=cur;last=(cx,cy);lost=0
   cv2.rectangle(fr,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3);cv2.circle(fr,(int(cx),int(cy)),12,(0,0,255),3)
   cv2.putText(fr,"26",(int(x1),max(24,int(y1)-7)),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2)
  elif selected is not None:
   lost+=1
   if last and lost<30:cv2.circle(fr,(int(last[0]),int(last[1])),12,(0,255,255),2)
  writer.write(fr);i+=1
 cap.release();writer.release()
 if selected is None:raise HTTPException(422,"선수 검출 실패: 선수가 잘 보이는 장면에서 몸 가운데를 지정하세요.")
 return {"job":j,"video_url":f"/result/{j}"}
@app.get("/result/{job}")
def result(job:str):
 p=D/f"{job}_tracked.mp4"
 if not p.exists():raise HTTPException(404,"결과 없음")
 return FileResponse(p,media_type="video/mp4",filename="football_ai_tracked.mp4")
