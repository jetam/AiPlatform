MyProject:

train models: python3 -m music.ml.train    // run from backend dir!

pip install torch numpy mido python-rtmidi


run servers with ./startServer.sh

        access frontend on localhost: file:///home/matej/code/MyProject/frontend/index.html

        run a http server in project dir (MyProject): python3 -m http.server 3000  // this is static file server!
                HTTP SERVER RUNNING ON PORT 3000

        server.py must be in MyProject so it sees frontend htmls

        for fastapi you need server: python3 -m uvicorn server:app --reload --port 5000 // runs server.py
                API server running on 5000


lsof -i -P -n | grep LISTEN // see which ports the server is listening on

connect with http: http://127.0.0.1:3000/frontend/index.html

api endpoints: http://127.0.0.1:5000/api/composer # todo: changed
api endpoints: http://127.0.0.1:5000/api/predictor


postgres:
CREATE USER matej WITH PASSWORD 'postgres12345';
psql -U matej -d myproject -h localhost 



Frontend
        ↓
Python API
        ↓
C++ logic
        ↓
Response → back to frontend


        try:
RNN , LSTM , transformer, 1D CNN