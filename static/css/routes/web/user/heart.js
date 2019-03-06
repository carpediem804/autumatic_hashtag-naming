var express = require('express');
var router = express.Router();

var userCtrl = require('../../../controllers/user.ctrl');

const sixHourMilliSec = 6 * 60 * 60 * 1000;
const monthMilliSec = 30 * 24 * 60 * 60 * 1000;


/*
	GET

	Read user.
*/
router.get('/', function(req, res, next) {
//  console.log('get user');

  res.render('user/heart');
});



module.exports = router;
