from flask import Flask, redirect, render_template, request
import caption_im
#import keras.backend.tensorflow_backend as tb
#tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)

@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/",methods=['POST'])
def captioning():
	if request.method=="POST":
		f = request.files['userfile']
		path = "./static/{}".format(f.filename)
		f.save(path)
		caption = caption_im.predict_caption(path)
		print(caption)

	return render_template("index.html",caption=caption)

if __name__=="__main__":
	app.run(debug=True)