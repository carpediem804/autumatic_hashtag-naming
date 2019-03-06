var express = require('express');
var router = express.Router();

/*
  POST

  matching page
*/
router.post('/', function(req, res, next) {
  var nickname = req.body.nickname;
  var topicName = req.body.topicName;

  res.render('matching/index', {
    nickname: nickname,
    topicName: topicName
  });
});

module.exports = router;
