var express = require('express');
var router = express.Router();

/*
  GET

  main page
*/
router.get('/', function(req, res, next) {
  var resultObject = new Object({});

  res.render('user/main');
});


module.exports = router;
