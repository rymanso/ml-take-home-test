## Instructions to run

### Setup

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export FLASK_APP=predict
flask run
```

### Root

`http://localhost:5000` has some visualisations of the data

### Query

Hit the endpoint:

`http://localhost:5000?tweet=hello`

Replace hello with any message you like to get a category out

Negative: 0
Positive: 1
Neutral: 2
Unknown: 3
