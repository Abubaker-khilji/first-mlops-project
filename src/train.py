#training source code
import mlflow
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


logging.basicConfig(level=logging.info,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    handlers=[logging.StreamHandler()
                              ,logging.fileHandler('mlops.log')])
                    
logging.info('Logging initialized')
logging.info('Starting training script')                    

iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)

logging.info('Data loaded successfully train and test is done too')
with mlflow.start_run():
    logging.info('training the randomforest model ......')
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    logging.info('Model trained successfully')

    predictions = model.predict(X_test)
    accuracy_score = accuracy_score(y_test,predictions)
    print('Accuracy:',accuracy_score)

    mlflow.log_metric('accuracy',accuracy_score)
    logging.info(f'Accuracy logged {accuracy:4f}successfully')