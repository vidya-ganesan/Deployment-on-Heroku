#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,request,jsonify,render_template
import pickle
import json


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


@app.route('/')
def Home():
    return render_template('index.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    return render_template('index.html',prediction_text = " Employee salary should be ${}".format(prediction))


# In[5]:


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


# In[ ]:


if __name__ == "__main__":
    app.run(port=5000)


# In[ ]:





# In[ ]:





# In[ ]:




