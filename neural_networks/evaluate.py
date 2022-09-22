# Python Imports
import math
import random
import traceback
from pdb import set_trace

# Internal Imports
from itcs4156.util.eval import RunModel
from itcs4156.util.timer import Timer
from itcs4156.util.metrics import mse, accuracy
from itcs4156.datasets.DataPreparation import HousingDataPreparation
from itcs4156.datasets.DataPreparation import MNISTDataPreparation
from itcs4156.assignments.neural_networks.NeuralNetwork import NeuralNetwork
from itcs4156.assignments.neural_networks.NeuralNetworkRegressor import NeuralNetworkRegressor
from itcs4156.assignments.neural_networks.NeuralNetworkClassifier import NeuralNetworkClassifier
from itcs4156.assignments.neural_networks.train import HyperParametersAndTransforms as hpt

def rubric_regression(mse, max_score=40):
    thresh = 12.5
    if mse <= thresh:
        score_percent = 100
    elif mse is not None:
        score_percent = (thresh / mse) * 100
        if score_percent < 40:
            score_percent = 40
    else:
        score_percent = 20
    score = max_score * score_percent / 100.0

    return score

def rubric_classification(acc, max_score=40):
    score_percent = 0
    if acc >= 0.93:
        score_percent = 100
    elif acc >= 0.85:
        score_percent = 90
    elif acc >= 0.70:
        score_percent = 80
    elif acc >= 0.60:
        score_percent = 70
    elif acc >= 0.50:
        score_percent = 60
    elif acc >= 0.40:
        score_percent = 55
    elif acc >= 0.30:
        score_percent = 50
    elif acc >= 0.20:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()
    
    task_info = [
       dict(
            model=NeuralNetworkRegressor,
            name='NeuralNetworkRegressor',
            data=HousingDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(mse=mse),
            eval_metric='mse',
            rubric=rubric_regression,
            trn_score=9999,
            eval_score=9999,
            successful=False,
        ),
        dict(
            model=NeuralNetworkClassifier,
            name='NeuralNetworkClassifier',
            data=MNISTDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            rubric=rubric_classification,
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
    ]
    
    total_points = 0

    for info in task_info:
        task_timer =  Timer()
        task_timer.start()
        try:
            params = hpt.get_params(info['name'])
            model_kwargs = params.get('model_kwargs', {})
            data_prep_kwargs = params.get('data_prep_kwargs', {})
            
            run_model = RunModel(info['model'], model_kwargs)
            data = info['data'](**data_prep_kwargs)
            X_trn, y_trn, X_vld, y_vld = data.data_prep(**info['data_prep'])

            trn_scores = run_model.fit(X_trn, y_trn, info['metrics'], pass_y=True)
            eval_scores = run_model.evaluate(X_vld, y_vld, info['metrics'], prefix=eval_stage.capitalize())
            
            if not math.isnan(trn_scores[info['eval_metric']]):
                info['trn_score'] = trn_scores[info['eval_metric']]
            if not math.isnan(eval_scores[info['eval_metric']]):
                info['eval_score'] = eval_scores[info['eval_metric']]
            
            info['successful'] = True
                
        except Exception as e:
            track = traceback.format_exc()
            print("The following exception occurred while executing this test case:\n", track)
        task_timer.stop()
        
        print("")
        points = info['rubric'](info['eval_score'])
        print(f"Points Earned: {points}")
        total_points += points

    print("="*50)
    print('')
    main_timer.stop()

    successful_tests = summary(task_info)
    final_mse, final_acc = get_eval_scores(task_info)
    total_points = int(round(total_points))
    
    print(f"Tests passed: {successful_tests}/{ len(task_info)}, Total Points: {total_points}/80\n")
    print(f"Final {eval_stage.capitalize()} MSE: {final_mse}")
    print(f"Final {eval_stage.capitalize()} Accuracy: {final_acc}")

    return total_points, main_timer.last_elapsed_time, final_mse, final_acc

def summary(task_info):
    successful_tests = 0

    for info in task_info:
        if info['successful']:
            successful_tests += 1
    
    if successful_tests == 0:
        return successful_tests

    return successful_tests

def get_eval_scores(task_info):
    return [i['eval_score'] for i in task_info]

if __name__ == "__main__":
    run_eval()


