# -*- coding: utf-8 -*-
'''
'''
import time, os
from loguru import logger as log
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError, HTTPException
import uvicorn, traceback
from pydantic import BaseModel, Field
from typing import Any
from common_table import main
from common.exceptions import ParsingError
from fastapi import UploadFile, File

app=FastAPI(
    title="tableRec",
    version="v3.3",
)



class MyResponse(BaseModel):
    isSuc: bool = Field(..., example=True)
    code: int = Field(..., example=0)
    msg: str = Field(..., example="success")
    res: Any = Field(...)


@app.post('/tableRec', summary="tableRec", response_model=MyResponse)
async def interface(file: UploadFile=File(...)):
    start = time.time()
    file_stream = await file.read()
    result = main(file_stream, stream = True)
    end = time.time()
    return MyResponse(isSuc=True, code=0, msg="{0:.2f}s".format(end - start), res=result)


@app.get("/downloadFile", summary = "下载AI处理后的Excel文件", description = "仅支持下载xls的Excel文件")
async def downloadFile():
    file_path = r'test/result.xls'
    return FileResponse(file_path, filename = 'result.xls')

@app.get("/downloadHtml", summary = "下载AI可视化后的Html", description = "")
async def downloadHtml():
    file_path = r'test/result.HTML'
    return FileResponse(file_path, filename = 'result.HTML')

@app.get("/downloadImage", summary = "下载AI可视化后的图像", description = "")
async def downloadFile():
    file_path = r'test/result_ceil.png'
    return FileResponse(file_path, filename = 'result_ceil.png')

@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers['X-Process-Time'] = str(process_time)
    return response

# API接口，传入参数检验失败
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error(traceback.format_exc())
    # print(str(exc.body))
    return JSONResponse(
        status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
        content={"isSuc":False,"code":1,"msg":"Missing or wrong parameter.","res": {}}
    )

# 图片解析失败
@app.exception_handler(ParsingError)
async def image_parser_error_exception_handler(request: Request, exc: ParsingError):
    log.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
        content={"isSuc":False,"code":exc.code,"msg":exc.message,"res":{}}
    )

# 404 Not Found
@app.exception_handler(HTTPException)
async def exception_handler(request: Request, exc: HTTPException):
    log.error(traceback.format_exc())
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
      content={'isSuc':False,'code':3,'msg':'Not Found','res': {}})

# 预料之外的错误
@app.exception_handler(Exception)
async def unexcept_exception_handler(request: Request, exc: Exception):
    log.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"isSuc":False,"code":4,"msg":"Internal Error","res":{}}
    )


if __name__ == '__main__':
    uvicorn.run(app='api:app', host="0.0.0.0", port=8018) #, workers = None, reload=True, debug=True)

