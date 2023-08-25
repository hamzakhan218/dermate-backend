
import numpy as np
from flask_cors import CORS

from flask_pymongo import PyMongo
from flask import Flask, request, send_from_directory, jsonify 
import cv2
import csv



import tensorflow as tf
class_names = ['Dry_skin', 'Normal_skin', 'Oily_skin']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print('Loading model ...')



def load_model():
    global  model, acne_type_model
     # Load the pre-trained skin type detection model
    model = tf.keras.models.load_model('D:/fyp/skin/saved_model/my_model')
 
    acne_type_model = tf.keras.models.load_model(
        'D:/fyp/DERMATE/dermate/acne detection/saved_model/my_model')

    print("Model Loaded!")


load_model()

app = Flask(__name__)
CORS(app)


# 




def get_products(skin_type, wrinkles, blemishes, acne_level, scars, none):

    # Define a dictionary of recommended products based on the selected skin concerns
    recommended_products = {
        'oily': {
            'wrinkles': 'Retinol',
            'blemishes': 'Salicylic Acid',
            'wrinkles_and_blemishes': 'Niacinamide',
            'scars': 'Vitamin C',
            'acne': 'Benzoyl Peroxide'
        },
        'dry': {
            'wrinkles': 'Hyaluronic Acid',
            'blemishes': 'Glycolic Acid',
            'wrinkles_and_blemishes': 'Vitamin C',
            'scars': 'Rosehip Oil',
            'acne': 'Tea Tree Oil'
        },
        'normal': {
            'wrinkles': 'Peptides',
            'blemishes': 'Benzoyl Peroxide',
            'wrinkles_and_blemishes': 'Azelaic Acid',
            'scars': 'Vitamin C',
            'acne': 'Salicylic Acid'
        }
    }
    
    # Create a list of recommended products based on the selected skin concerns
    recommended_products_list = []
    
    if wrinkles:
        recommended_products_list.append(recommended_products[skin_type]['wrinkles'])
    if blemishes:
        recommended_products_list.append(recommended_products[skin_type]['blemishes'])
    if none:
        recommended_products_list.append(recommended_products[skin_type]['none'])
    if scars:
        recommended_products_list.append(recommended_products[skin_type]['scars'])
    if acne_level == 'mild':
        recommended_products_list.append(recommended_products[skin_type]['acne'])
        
    return recommended_products_list

def predict_skin_type(face_img):
    class_names1 = ['Dry', 'Combination', 'Oily']

   

    # Resize the image to the input shape expected by the model
    new_image = tf.image.resize(face_img, [224, 224])

    # Convert the image to a numpy array and normalize the pixel values
    new_image = tf.keras.preprocessing.image.img_to_array(new_image)
    new_image = new_image / 255.0

    # Add a batch dimension to the input image
    new_image = np.expand_dims(new_image, axis=0)

    # Make a prediction using the pre-trained model
    pred1 = model.predict(new_image)

    # Get the predicted skin type class name
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    
    # Return the predicted skin type
    return pred_class1


# # Register a new user
# @app.route('/register', methods=['POST'])
# def register():
#     try:
#         data = request.get_json()
#         email = data['email']
#         password = data['password']

#         # Check if the user with the same email already exists
#         existing_user = mongo.db.users.find_one({'email': email})
#         if existing_user:
#             return jsonify({'message': 'Email is already registered'}), 400

#         # Create a new user document
#         new_user = {
#             'email': email,
#             'password': password
#         }

#         # Insert the new user document into the MongoDB collection
#         mongo.db.users.insert_one(new_user)

#         return jsonify({'message': 'User registered successfully'}), 201

#     except Exception as e:
#         print('Error occurred during registration:', e)
#         return jsonify({'message': 'Internal server error'}), 500



@app.route('/api/products/<skin_type>')
def get_filtered_products(skin_type):
    filtered_products = []

    with open('result.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['skin_type'] == skin_type:
                filtered_products.append({
                    'label': row['label'],
                    'url': row['url'],
                    'brand': row['brand'],
                    'name': row['name'],
                   
                    'image': row['image'],
                    'spf':row['spf'],
                    'concern'  :row['concern']  ,      
                    'concern2' : row['concern2'],
                    'concern3' : row['concern3'],
                    'formulation':row['formulation'],
                    'key_ingredient': row['key_ingredient'],
                    'url': row['url'] 
                })

    return jsonify(filtered_products)




# 
@app.route('/api/products')
def get_recommended_products():
    # Get the skin concerns from the query parameters
    skin_type = request.args.get('skinType')
    wrinkles = request.args.get('wrinkles')
    blemishes = request.args.get('blemishes')
    none = request.args.get('none')
    scars = request.args.get('scars')
    acne_level = request.args.get('acne_level')
   
    

    # Call a function to get the recommended products based on the skin concerns
    recommended_products = get_products(skin_type, wrinkles, blemishes, none, scars, acne_level)

    # Return the recommended products as a JSON object
    return jsonify({'products': recommended_products})


###########################

@app.route('/api/submitFeedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        name = data['name']
        email = data['email']
        gender = data['gender']
        review = data['review']
        sentiment = data['sentiment']

        # Write the feedback to a CSV file (rating.csv)
        with open('rating.csv', 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([name, email, gender, review,sentiment])

        return jsonify({'message': 'Feedback submitted successfully'}), 201

    except Exception as e:
        print('Error occurred during feedback submission:', e)
        return jsonify({'message': 'Internal server error'}), 500



##############

@app.route('/api/feedback', methods=['GET'])
def get_feedback_data():
    feedback_data = []
    try:
        with open('rating.csv', 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                name, email, gender, review, sentiment = row
                feedback_data.append({
                    'name': name,
                    'email': email,
                    'gender': gender,
                    'review': review,
                    'sentiment' : sentiment
                    
                })
        return jsonify({'feedback': feedback_data}), 200
    except FileNotFoundError:
        return jsonify({'message': 'Feedback data not found'}), 404
    except Exception as e:
        return jsonify({'message': 'Error occurred while fetching feedback data'}), 500




######
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        file.save("D:/fyp/DERMATE/dermate/flaskApi/folder/" + file.filename)
        return {"message": "File uploaded successfully", "file_url": f"http://localhost:5000/images/{file.filename}"}
    else:
        return {"error": "No file found"}


@app.route("/images/<filename>")
def image(filename):
    return send_from_directory("D:/fyp/DERMATE/dermate/flaskApi/folder/", filename)

@app.route("/helloworld", methods=["GET"])
def hello_world():
    return "Hello World"

@app.route("/", methods=["GET"])
def root():
    return "This is the home route"


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file:
        file.save("D:/fyp/DERMATE/dermate/flaskApi/folder/" + file.filename)
        image_path = "D:/fyp/DERMATE/dermate/flaskApi/folder/" + file.filename

        # Load the image file
        img = cv2.imread(image_path)
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # If a face is detected
        if len(faces) > 0:
            # Get the face region
            x, y, w, h = faces[0]
            face_img = img[y:y+h, x:x+w]
            # Pass the face region to your skin type detection code
            skin_type = predict_skin_type(face_img)
            # Make acne type prediction
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.
            acne_type_preds = acne_type_model.predict(image)
            acne_type = np.argmax(acne_type_preds)
            if acne_type == 0:
                acne_type = "Normal"
            elif acne_type == 1:
                acne_type = "Severe"
            else:
                acne_type = "Moderate"

            # Get the predicted probabilities
            pred_probabilities = []
            for pred_prob in acne_type_preds[0]:
                pred_probabilities.append(float(pred_prob))

            # Return the predicted skin type, acne type, and probabilities
            return {
                "message": "Prediction made successfully",
                "skin_type": skin_type,
                "acne_type": acne_type,
                "probabilities": pred_probabilities
            }
        else:
            return {"error": "No face detected."}
    else:
        return {"error": "No file found"}



if __name__ == '__main__':
    app.run(host='localhost', port=5000)
