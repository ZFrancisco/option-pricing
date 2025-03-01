from flask import Flask

app = Flask(__name__)

@app.route('/')
def home_page():
    try: 
        return  
    except Exception as e:
        return 
    
if __name__ == '__main__':
    app.run(debug=True)