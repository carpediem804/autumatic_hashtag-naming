var express = require('express');
var router = express.Router();

var userCtrl = require('../../../controllers/user.ctrl');

const sixHourMilliSec = 6 * 60 * 60 * 1000;
const monthMilliSec = 30 * 24 * 60 * 60 * 1000;


/*
	GET

	Read user.
*/
router.get('/signin', function(req, res, next) {
  console.log('get user');

  res.render('user/signin');
});


/*
	POST

	Create user.
*/
router.post('/', function(req, res, next) {
  var email = req.body.email;
  var password = req.body.password;
  var confirm = req.body.confirm;

  userCtrl.signup(email, password, "local", function(error, signupObject){

  	res.json(signupObject);
  });
});

/*
	PUT

	Update user.
*/
router.put('/', function(req, res, next) {
  var email = req.body.email;
  var password = req.body.password;

  var resultObject = new Object({});

  console.log("Update user");

  resultObject.test = true;

  res.json(resultObject);
});


/*
	DELETE

	Delete user.
*/
router.delete('/', function(req, res, next) {
  var email = req.body.email;
  var password = req.body.password;

	console.log("Delete user data");

	userCtrl.withdraw(email, function(error, withdrawObject){
		res.json(withdrawObject);
	});
});

/*
	DELETE

	Delete user.
*/
router.post('/withdraw', function(req, res, next) {
  var email = req.body.email;
  var password = req.body.password;

	console.log("Delete user data");

	userCtrl.withdraw(email, function(error, withdrawObject){
		res.json(withdrawObject);
	});
});


/*
	POST

	Try user signin.
*/
router.post('/signin/:platformName?', function(req, res, next) {
	var platformName = req.params.platformName || "local";
	var email = req.body.email.trim();
	var password = req.body.password;

  userCtrl.signin(email, password, platformName, function(error, signinObject){
		if(signinObject.signin){
			// signin success
			const accessToken = signinObject.accessToken;
			const refreshToken = signinObject.refreshToken;

			res.cookie('access_token', accessToken, { expires: new Date(Date.now() + sixHourMilliSec), httpOnly: true });
			res.cookie('refresh_token', refreshToken, { expires: new Date(Date.now() + monthMilliSec), httpOnly: true });
			res.json(signinObject);

		}else{
			// signin fail
			res.json(signinObject);
		}
	});
});

/*
	POST

	Try user signout.
*/
router.post('/signout', function(req, res, next) {
	var email = req.decoded.data.email;
	console.log("signout");

	userCtrl.signout(email, function(error, resultSignout){
    if(resultSignout.signout){
  		res.clearCookie("access_token");
  		res.clearCookie("refresh_token");
    }
    res.json(resultSignout);
	});


});


/*
	POST

	Try user signup and signin
*/
router.post('/signup', function(req, res, next) {
	var email = req.body.email.trim();
	var password = req.body.password;
	var confirm = req.body.confirm;


  userCtrl.signupAndSignin(email, password, confirm, function(error, resultObject){
    if(resultObject.signup){
      const accessToken = resultObject.accessToken;
      const refreshToken = resultObject.refreshToken;

  		res.cookie('access_token', accessToken,{ expires: new Date(Date.now() + sixHourMilliSec), httpOnly: true });
  		res.cookie('refresh_token', refreshToken,{ expires: new Date(Date.now() + monthMilliSec), httpOnly: true });

      res.render('user/signin_success');
    }else{
      res.json(resultObject);
    }

  });

});



module.exports = router;
