## About

[![Build Status](https://travis-ci.org/vinitkumar/json2xml.svg?branch=master)](https://travis-ci.org/vinitkumar/json2xml)

A Simple python utility to convert JSON to XML(Supports 3.5.x and 3.6.x).
It can be used to convert a json file to xml or from an URL that returns json data.

### How to install

```
pip3 install json2xml
```

### Usage

### Command Line

```
python -m src.cli --file="examples/example.json"
python -m src.cli --url="https://coderwall.com/vinitcool76.json"
python -m src.cli --data '{"login":"mojombo","id":1,"avatar_url":"https://avatars0.githubusercontent.com/u/1?v=4"}'
```

### Inline in Code

#### From a file

```python
from src.json2xml import Json2xml
data = Json2xml.fromjsonfile('examples/example.json').data
data_object = Json2xml(data)
data_object.json2xml() #xml output
```

#### From an URL

```python
from src.json2xml import Json2xml
data = Json2xml.fromurl('https://coderwall.com/vinitcool76.json').data
data_object = Json2xml(data)
data_object.json2xml() #xml output
```

#### From JSON string

```python
from src.json2xml import Json2xml
data = Json2xml.fromstring('{"login":"mojombo","id":1,"avatar_url":"https://avatars0.githubusercontent.com/u/1?v=4"}').data
data_object = Json2xml(data)
data_object.json2xml() #xml output
```

### Bugs, features

- If you find any bug, open a [new ticket](https://github.com/vinitkumar/json2xml/issues/new)
- If you have an intresting Idea for contribution, open a ticket for that too and propose the idea. If we agree, you can open a pull request.
