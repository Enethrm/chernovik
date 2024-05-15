import uvicorn

from fastapi import FastAPI
from colorization.colorization_handler import colorization_router
from super_resolution.super_resolution_handler import super_resolution_router
from denoising.denoising_hanler import denoising_router


app = FastAPI()
app.include_router(colorization_router)
app.include_router(super_resolution_router)
app.include_router(denoising_router)

@app.get('/ping')
async def main():
    return 200


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000)