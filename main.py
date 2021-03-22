from typing import List
from fastapi import FastAPI,File ,Request
from fastapi.datastructures import UploadFile
from starlette.responses import Response
import uvicorn
import shutil
import MODEL
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


UPLOAD_FOLDER="/root/SignatureApp/static/images/"
similar = True 
similarityScore=1

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name ='static')

@app.get("/home",response_class=HTMLResponse)
async def home(requests:Request ):
    return templates.TemplateResponse("index.html",{"request":requests})

@app.post("/upload",response_class=HTMLResponse)
async def home(requests:Request, imagea: UploadFile = File(...), imageb: UploadFile = File(...)):   #name must be similar to that of index.html
    with open(UPLOAD_FOLDER+imagea.filename,"wb") as buffer:
        shutil.copyfileobj(imagea.file,buffer)
    with open(UPLOAD_FOLDER+imageb.filename,"wb") as buffer:
        shutil.copyfileobj(imageb.file,buffer)
    similar,similarityScore = getSimilarity(UPLOAD_FOLDER+imagea.filename,UPLOAD_FOLDER+imageb.filename)
    return templates.TemplateResponse("index.html",{"request":requests, "prediction":similar ,"similarity":similarityScore})

pred ={}

@app.post("/getsimilarity")
async def create_upload_files(images: List[UploadFile] = File(...)):
    for img in images:
      with open(UPLOAD_FOLDER+img.filename,"wb") as buffer:
            shutil.copyfileobj(img.file,buffer)
    for img in images:
        idx = images.index(img)+1
        while idx < len(images):
            label = (img.filename+","+images[idx].filename)
            similar,similarityScore = getSimilarity(UPLOAD_FOLDER+img.filename,UPLOAD_FOLDER+images[idx].filename)
            pred[label]=({"similar":str(similar),"similarityScore":str(similarityScore)})
            idx += idx     
    return pred

def getSimilarity(image1, image2):
    res,d = MODEL.predict(image1,image2, 0.7, simModel, MODEL.data_generator, verbose=False)
    return res, d
    
if __name__ == "__main__":
    simModel = MODEL.Siamese()
    simModel.init_from_file("/root/SignatureApp/model/model.pkl.gz")
    uvicorn.run(app,host='192.168.170.178',port=5000, debug=False)
