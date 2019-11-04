let express = require("express");
let morgan = require("morgan");
let uuidv4 = require('uuid/v4');
let bodyParser = require('body-parser');
const cors = require('cors');
var fs = require('fs');
var mongoose = require('mongoose');
var Schema = mongoose.Schema;
var multer = require('multer');
let app = express();

app.use(morgan("dev"));
app.use(cors());

var jsonParser = bodyParser.json();

app.get("/accounts", (req, res, next) => {
    res.status(200).json(getAllAccounts());
});

let accountFields = [
    "plate", "face_id", "name"
];

var Item = new ItemSchema(
{ img: 
    { data: Buffer, contentType: String }
}
);
var plateItem = mongoose.model('Plates',ItemSchema);
var faceItem = mongoose.model('Faces',ItemSchema);

// Expecting data in body.
app.post("/accounts", jsonParser, (req, res) => {
    console.log(req.body);
    var jsonObject = req.body;
    var validObject = true;
    var missingFields = [];

    accountFields.forEach(function(field) {
        if (jsonObject[field] === undefined) {
            validObject = false;
            missingFields.push(field);
        }
    });

    if (validObject) {
        var newItem = new Item();
        newItem.img.data = fs.readFileSync(req.files.userPhoto.path)
        newItem.img.contentType = 'image/png';
        newItem.save();

        insertAccount(validObject);
        res.status(201).json(jsonObject);
    } else {
        res.status(406).json(missingFields);
    }
});

app.delete("/accounts/:id", (req, res) => {
    let accountId = req.params.id;
    let removedAccount = removeAccount(accountId);
    
    if (foundId) {
        res.status(200).json(removedAccount);
    } else {
        res.status(404).json({message: "Post with id not found.",
                                status: "404"});
    }
});

app.put("/accounts/:id", jsonParser, (req, res) => {
    let accountId = req.body["id"];
    if (accountId === undefined) {
        res.status(406).json({message: "ID missing in request body",
                                status: "406"});
    } else {
        updateAccount(accountId, req.body);
    }
});

const MongoClient = require('mongodb').MongoClient;
const assert = require('assert');

// Connection URL
const url = 'mongodb://localhost:27017';

// Database Name
const dbName = 'accenturecv_db';

// Create a new MongoClient
const client = new MongoClient(url);

app.listen("8080", () => {
    console.log("Listening on port 8080");

    // Use connect method to connect to the Server
    client.connect(function(err) {
        assert.equal(null, err);
        console.log("Connected successfully to server");
    
        const db = client.db(dbName);
    
        client.close();
    });

    mongoose.connect(url);
});

app.use(multer({ dest: './uploads/',
    rename: function (fieldname, filename) {
        return filename;
    },
}));