var express = require('express');
var router = express.Router();

/*
  GET

  chatting page
*/
router.get(['/', '/:nickname/:topicName'], function(req, res, next) {
  var nickname = req.params.nickname;
  var topicName = req.params.topicName || "public";
  res.render('chatting/index', {
    nickname: nickname,
    topicName: topicName
  });
});

module.exports = router;
