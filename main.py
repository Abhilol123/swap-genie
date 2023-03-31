import logging
from fastapi import FastAPI


def isServerLive():
    return True


app = FastAPI()


@app.on_event("startup")
async def startup():
    logging.info("started server!")


@app.on_event("shutdown")
async def shutdown():
    logging.info("stoped server!")
