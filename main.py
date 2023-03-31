import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from multiprocessing import Manager
from controller.inference import inferenceController


app = FastAPI()


manager = Manager()
store = manager.dict()
store["model_running"] = False


@app.post("/inference")
async def inference_stable_diffusion(request: Request):
    try:
        logging.info("running POST /inference")
        while (store["model_running"]):
            time.sleep(1)
        store["model_running"] = True
        data = await inferenceController(request)
        store["model_running"] = False
        if data == None:
            return JSONResponse(content={"message": "could not get any images."}, status_code=404)
        else:
            return data
    except Exception as e:
        store["model_running"] = False
        logging.error(str(e))
        return JSONResponse(content={"message": "internal server error"}, status_code=500)


@app.on_event("startup")
async def startup():
    logging.info("started server!")


@app.on_event("shutdown")
async def shutdown():
    logging.info("stoped server!")
