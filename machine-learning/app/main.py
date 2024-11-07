
import asyncio
import gc
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Iterator, List, Optional
from zipfile import BadZipFile
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn
import base64
import numpy as np
from tempfile import NamedTemporaryFile
import cv2
# import zmq
# import requests
# import json
import pandas as pd
import shutil
from pathlib import Path
import io

from os.path import dirname, realpath, join
import sys
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)
sys.path.insert(0, P_DIR)

import orjson
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import ORJSONResponse, JSONResponse
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile
from starlette.formparsers import MultiPartParser

from .models.base import InferenceModel

from app.config import PreloadModelData, log, settings
from app.mqtt import MqttClient
from .models.cache import ModelCache
from app.schemas import (
    MessageResponse,
    ModelType,
    TextResponse,
    AddManyResponse,
    GetResponse, 
    SearchResponse,
    QueryResponse,
    DataFrameResponse
    
)


from facedb import FaceDB


MultiPartParser.max_file_size = 2**26  # spools to disk if payload is 64 MiB or larger

model_cache = ModelCache(revalidate=settings.model_ttl > 0)
thread_pool: ThreadPoolExecutor | None = None
lock = threading.Lock()
active_requests = 0
last_called: float | None = None

# context = zmq.Context()
# socket = context.socket(zmq.SUB)
# socket.connect("tcp://devcontainer:5555")  # Connect to the publisher socket of the detector
# socket.subscribe(b"")  # Subscribe to all topics

mqtt_client = None
face_db = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global thread_pool, mqtt_client, face_db
    log.info(
        (
            "Created in-memory cache with unloading "
            f"{f'after {settings.model_ttl}s of inactivity' if settings.model_ttl > 0 else 'disabled'}."
        )
    )

    try:
        if settings.request_threads > 0:
            # asyncio is a huge bottleneck for performance, so we use a thread pool to run blocking code
            thread_pool = ThreadPoolExecutor(settings.request_threads) if settings.request_threads > 0 else None
            log.info(f"Initialized request thread pool with {settings.request_threads} threads.")
        if settings.model_ttl > 0 and settings.model_ttl_poll_s > 0:
            asyncio.ensure_future(idle_shutdown_task())
        if settings.preload is not None:
            await preload_models(settings.preload)

        # load facedb
        face_db = await setup_face_db()

        # Initialize and start the MQTT client
        mqtt_client = MqttClient()
        mqtt_client._start()  # Connect to MQTT broker
            
        ## Automatically start the recognize_face task
        # asyncio.create_task(recognize_face())
        
                
        yield
    finally:
        log.handlers.clear()
        for model in model_cache.cache._cache.values():
            del model
        if mqtt_client is not None:
            mqtt_client.stop()
        if thread_pool is not None:
            thread_pool.shutdown()
        gc.collect()

async def preload_models(preload_models: PreloadModelData) -> None:
    log.info(f"Preloading models: {preload_models}")
    if preload_models.clip is not None:
        await load(await model_cache.get(preload_models.clip, ModelType.CLIP))
    if preload_models.facial_recognition is not None:
        await load(await model_cache.get(preload_models.facial_recognition, ModelType.FACIAL_RECOGNITION))

async def setup_face_db() -> FaceDB:
    log.info("Setting up FaceDB")
    return FaceDB(
        path=join(P_DIR, "facedata"),
        metric="cosine",  # hoặc cosine  euclidean
        database_backend="chromadb",
        embedding_dim=512,
        module="arcface",  # hoặc face_recognition
    )


def update_state() -> Iterator[None]:
    global active_requests, last_called
    active_requests += 1
    last_called = time.time()
    try:
        yield
    finally:
        active_requests -= 1


app = FastAPI(
    lifespan=lifespan,
    title="SysEye API",
    description="API for managing and recognizing faces using vector database",
    version="1.0.0",
)

@app.get("/", response_model=MessageResponse)
async def root() -> dict[str, str]:
    return {"message": "{}".format(pd.DataFrame([(f.name, f.embedding) for f in face_db.all(include=["embedding"])]))}

@app.get("/ping", response_model=TextResponse)
def ping() -> str:
    log.info("__call ping__")
    return TextResponse(name="pong")

####################################################
# async def recognize_face():
#     try:
#         log.info("call from face recognition from events and snapshot....")
#         mqtt_client.subscribe(recognize_snapshot)  # mqtt_client.publish_to_mqtt_topic call_recognize_api ,
#         mqtt_client.subscribe(recognize_events)
#     except Exception as ex:
#         log.exception(ex)
#         # await asyncio.sleep(5)
#         time.sleep(5)

# @app.post("/start_consumer/")
# async def start_consumer(background_tasks: BackgroundTasks):
#     background_tasks.add_task(recognize_face)  # receive_frames
#     return {"message": "Consumer started successfully"}



@app.post("/predict", dependencies=[Depends(update_state)])
async def predict(
    model_name: str = Form(alias="modelName"),
    model_type: ModelType = Form(alias="modelType"),
    options: str = Form(default="{}"),
    text: str | None = Form(default=None),
    image: UploadFile | None = None,
) -> Any:
    if image is not None:
        inputs: str | bytes = await image.read()
    elif text is not None:
        inputs = text
    else:
        raise HTTPException(400, "Either image or text must be provided")
    try:
        kwargs = orjson.loads(options)
    except orjson.JSONDecodeError:
        raise HTTPException(400, f"Invalid options JSON: {options}")

    model = await load(await model_cache.get(model_name, model_type, ttl=settings.model_ttl, **kwargs))
    model.configure(**kwargs)
    outputs = await run(model.predict, inputs)
    return ORJSONResponse(outputs)

###################################


# Endpoint để nhận diện khuôn mặt đã biết names: Optional[List[str]] = None, 
@app.post("/recognize", response_model=GetResponse)
async def recognize(images: UploadFile = File(None), image: UploadFile | None = None):
    try:
        if images is not None:
            img = await images.read()
            nparr = np.frombuffer(img, np.uint8)
            inputs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image is not None:
            inputs: str | bytes = await image.read()
        else:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        if inputs is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Perform face recognition
        result = face_db.recognize(img=inputs, include=["name", "id", "distance" "embedding"])
        if not result:
            raise HTTPException(status_code=404, detail="Face not recognized")
        
        return GetResponse(
            name=result["name"],
            id=result["id"],
            # boundingBox= result["boundingBox"],
            distance=result["distance"],
            embedding=result["embedding"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_face/")
async def add_face(
    name: str,
    img: UploadFile= File(None),
    embedding: Optional[list] = None,
    id: Optional[str] = None,
    check_similar: bool = True,
    save_just_face: bool = True
):
    try:
        img_file = None
        
        # If an image file is provided, save it to a temporary file
        if img is not None:
            with NamedTemporaryFile(delete=False) as temp_file:
                shutil.copyfileobj(img.file, temp_file)
                temp_file_path = temp_file.name
                img_file = open(temp_file_path, "rb")
                
                # Read the content of the file
                img_bytes = img_file.read()
                
        # Call the add method with the provided parameters
        idx = face_db.add(
            name=name,
            img=img_bytes if img_bytes else None,
            embedding=embedding,
            id=id,
            check_similar=check_similar,
            save_just_face=save_just_face
        )

        # Close the file if it was opened
        if img_file is not None:
            img_file.close()
        
        return {"id": idx}
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_many", response_model=AddManyResponse)
async def add_many(
    files: List[UploadFile] = File(...),
    names: Optional[str] = None, 
    check_similar: bool = True,
    save_just_face: bool = True
):
    imgs = []
    names_list = []
    names = names.split(",")
    for idx, file in enumerate(files):
        temp_file_path = Path("/tmp") / file.filename
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        imgs.append(temp_file_path)
        
        # Kiểm tra xem người dùng đã cung cấp tên cho ảnh thứ idx hay chưa
        if names and idx < len(names):
            names_list.append(names[idx])
        else:
            names_list.append(Path(file.filename).stem)
    
    # Sử dụng danh sách tên của tệp
    ids, failed_indexes = face_db.add_many(imgs=imgs, names=names_list, check_similar=check_similar)
    log.debug(ids)
    
    return AddManyResponse(ids=ids, failed_indexes=failed_indexes)



# Endpoint để cập nhật thông tin
@app.put("/update", response_model=GetResponse)
async def update(id: str, name: Optional[str], image: UploadFile= File(None)):
    try:
        # Read image from file and convert to np.ndarray
        img = await image.read()
        nparr = np.frombuffer(img, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Failed to decode image file")
        face_db.update(id = id,name = name, img = img_array)
        
        return GetResponse(name=name, id=id, img="Updated", distance=None, embedding=None)
    except cv2.error as e:
        raise HTTPException(status_code=400, detail=f"OpenCV error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint để lấy thông tin
@app.get("/get", response_class=HTMLResponse)
async def get(id: str):
    result = face_db.get(id=id, include=["name", "img"])
    if not result:
        raise HTTPException(status_code=404, detail="Record not found")
    img_base64 = GetResponse.array_to_base64(result["img"])
    image_data = base64.b64decode(img_base64)
    data =  GetResponse(name=result["name"], id = id, img = img_base64)
    return Response(content=image_data, media_type="image/jpeg")
    # return JSONResponse(content={"name": data.name, "id": data.id, "distance": data.distance, "embedding": data.embedding, "img": f"/images/{data.img}"})

@app.get("/get_all", response_model=None)
async def get_all(response: Response):
    result = face_db.all(include=["name", "id"])
    if not result:
        raise HTTPException(status_code=404, detail="Record not found")

    # Tạo DataFrame từ kết quả
    df = result.df[["name", "id"]]

    # Chuyển đổi DataFrame thành dạng dữ liệu và cột cho schema
    columns = df.columns.tolist()
    data = df.values.tolist()   

    # Tạo một đối tượng của lớp DataFrameResponse từ dữ liệu và cột
    data_response = DataFrameResponse(data=data, columns=columns)

    # Tạo nội dung CSV từ DataFrame và gửi lại cho người dùng để tải xuống
    csv_content = df.to_csv(index=False)
    response.headers["Content-Disposition"] = "attachment; filename=data.csv"
    response.headers["Content-Type"] = "text/csv"
    response.content = csv_content

    return data_response

@app.get("/count")
async def count():
    result = face_db.count()
    if not result:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return {"count": result}


# Endpoint để xóa thông tin
@app.delete("/delete")
async def delete(id: str):
    face_db.delete(id=id)
    return {"message": "Delete successful"}

# Endpoint để tìm kiếm thông tin
@app.post("/search", response_model=SearchResponse)
async def search(file: UploadFile = File(...)):
    temp_file_path = Path("/tmp") / file.filename
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    
    embedding = face_db.embedding_func(str(temp_file_path))
    results = face_db.search(embedding=embedding, include=["name"])
    if not results:
        raise HTTPException(status_code=404, detail="No matching records found")
    
    return SearchResponse(results=[GetResponse(name=res["name"]) for res in results])

# Endpoint để truy vấn thông tin
@app.get("/query", response_model=QueryResponse)
async def query(name: str):
    results = face_db.query(name=name, include=["name", "id", "embedding"])
    if not results:
        raise HTTPException(status_code=404, detail="No matching records found")
    
    return QueryResponse(results=[GetResponse(name=res["name"], id=res["id"], embedding=res["embedding"]) for res in results])


###################################

async def run(func: Callable[..., Any], inputs: Any) -> Any:
    if thread_pool is None:
        return func(inputs)
    return await asyncio.get_running_loop().run_in_executor(thread_pool, func, inputs)


async def load(model: InferenceModel) -> InferenceModel:
    if model.loaded:
        return model

    def _load(model: InferenceModel) -> None:
        with lock:
            model.load()

    try:
        await run(_load, model)
        return model
    except (OSError, InvalidProtobuf, BadZipFile, NoSuchFile):
        log.warning(
            (
                f"Failed to load {model.model_type.replace('_', ' ')} model '{model.model_name}'."
                "Clearing cache and retrying."
            )
        )
        model.clear_cache()
        await run(_load, model)
        return model


async def idle_shutdown_task() -> None:
    while True:
        log.debug("Checking for inactivity...")
        if (
            last_called is not None
            and not active_requests
            and not lock.locked()
            and time.time() - last_called > settings.model_ttl
        ):
            model_cache.clear()
        await asyncio.sleep(settings.model_ttl_poll_s)

def cleanup() -> None:
    if mqtt_client is not None:
        mqtt_client.stop()
    if thread_pool is not None:
        thread_pool.shutdown()

def sigterm_handler(signal, frame) -> None:
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGTERM, sigterm_handler)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=3003)
    finally:
        cleanup()
