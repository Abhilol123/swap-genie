pip install -r requirements.txt;

mkdir -p videos;
mkdir -p images;
mkdir -p data;
mkdir -p lora;

uvicorn main:app;
