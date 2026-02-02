"""
THIS SCRIPT IS FOR DETECTING STRUCTURAL BREAKS AND ASSESING THE RECALL AND PRECISION
OF THESE FINDINGS WITH ROC CURVE BASED ON VARIOUS STATISTICAL TESTS.


PLEASE NOTE : 1) This script has been developed on my personal computer
and requires installing rapture library( Pip install rapture), its an
open source public library and available for installation

             2) This script runs perfectly fine producing  roc curves for 
                with out being assesed by tests mentioned below and 
                also with being assesed( corrected) by tests mentioned below.


Author: Ravi K Gudur

This file contains the scripts for simulating data with breaks and
identifying them using rapture module
These breaks are verified by following statistical test
1) Anderson Darling
2) Kullback-Leibler (KL) divergence
3) wasserstein_distance
4) Koglomorov test 

The idea is break the data into segments 
1) segemnts with original breaking points
2) segments with detected breaking points
3) these two segments are compared using above tests( if they are from same distribution)
4) whenever the test failed that particular segment is removes and that brake point is 
removed
5) original broken points are converted into binary along with rapture's detected broken points
6) rapture's detected brake points are modified as per tests( they are removed if test failed)
7) True_labels ( binary version of original brake points, break points are 1, rest are zero)
8) Predicted_lables(raptures break points turned binary)
9) Predicted_labels are modified as per test results 
10) True labels and predicted labels are used to find precision and recall and roc curve

"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import statsmodels.api as sm
import numpy as np
from scipy.stats import entropy
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from scipy.stats import anderson_ksamp
from sklearn.metrics import roc_curve, precision_recall_curve, auc



def simulate_data(n_samples,n_breaks,pre_break_mean=0.5, post_break_mean=0.8):

    """ Function to simulate Data"""
   
    np.random.seed(42)

    break_points = np.sort(np.random.choice(range(n_samples), size=n_breaks, replace=False))
        
    data = np.random.normal(pre_break_mean, 0.1, n_samples)
    data[break_points] = np.random.normal(post_break_mean, 0.1, n_breaks)
    
    original_segments = []
    start_idx = 0
    
    # Iterate through breakpoints
    for breakpoint in break_points:
        segment = data[start_idx:breakpoint]
        original_segments.append(segment)
        start_idx = breakpoint
    
    # Add the last segment
    if len(data[start_idx:]):
        original_segments.append(data[start_idx:])
    
    true_labels = np.zeros(n_samples)
    true_labels[break_points] = 1

    plt.figure(figsize=(8, 4))
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Data')
    plt.title('Simulated Data with Structural Breaks')
    plt.show()

    return data,original_segments,true_labels


def detect_breaks(data,n_bkps,n_samples):
    """ break detection"""
   
    detected_segments=[]
    # Break point detection
    model = "l2"  # model used (l2-norm)
    algo = rpt.Dynp(model=model).fit(data)
    result = algo.predict(n_bkps)  # Specify the expected number of change points
    
    # Convert break points to predicted labels
    predicted_labels = np.zeros(n_samples)
    for bp in result:
        if bp < n_samples:
            predicted_labels[bp] = 1


    detected_segments = []
    start_idx = 0
    
    # Iterate through breakpoints
    for breakpoint in result:
        segment = data[start_idx:breakpoint]
        detected_segments.append(segment)
        start_idx = breakpoint
    
    # Add the last segment
    if len(data[start_idx:]):
        detected_segments.append(data[start_idx:])

    # Visualize the results
    rpt.display(data, result)
    plt.title("break detection")
    plt.show()

    return predicted_labels, detected_segments, result


# tests for comparing two segememts

def kl_test(segment1, segment2, threshold=0.1):
    """
    Kullback-Leibler (KL) divergence between their empirical distributions.
    """
    # empirical distributions
    dist1 = np.histogram(segment1, bins='auto', density=True)[0]
    dist2 = np.histogram(segment2, bins='auto', density=True)[0]
    
    # KL divergence between distributions
    if len(dist1)== len(dist2):
        kl_divergence_1_to_2 = entropy(dist1, dist2)
        kl_divergence_2_to_1 = entropy(dist2, dist1)
    else:
        return 0
    # Decision
    if kl_divergence_1_to_2 < threshold and kl_divergence_2_to_1 < threshold:
        print("Segments are likely from the same distribution")
        return 1
    else:
        print("Segments are not likely from same distributions")
        return 0
    

def wd_test(segment1, segment2):
    """.
    wasserstein_distance to determine if segements are from same distribution
    """
    # empirical distributions
    dist1 = np.histogram(segment1, bins='auto', density=True)[0]
    dist2 = np.histogram(segment2, bins='auto', density=True)[0]
    
    # Wasserstein distance between the empirical distributions
    distance = wasserstein_distance(dist1, dist2)
    
    # Decision
    if distance < 0.05: 
        print("Segments are likely from the same distribution")
        return 1
    else:
        print("Segments are  not likely from same distributions")
        return 0
    

def ks_test(segment1, segment2, alpha=0.05):
    """
    Kolmogorov-Smirnov (KS) test to compare two data segments.
    """
    p_value = ks_2samp(segment1, segment2).pvalue
    
    if p_value < alpha:
        print(" segments are not likely from same distributions")
        return 0
    else:
        print(" segments are likely from the same distribution")
        return 1
    

def anderson_test(segment1, segment2, alpha=0.05):
    """
    Anderson-Darling test.
    """
    p_value = anderson_ksamp([segment1, segment2]).pvalue
    
    if p_value < alpha:
        print("Segments are not likely from same distribution")
        return 0
    else:
        print("Segments are likely from the same distribution")
        return 1
    

def test_based_labels(detected_segments, original_segments,predicted_breaks,predicted_labels,true_labels,n_samples):

    """
    This function calculates recalls and precisions based on test and also
    plots ROC curve.
    """
            
    result_dict ={}
    for i in range(len(detected_segments)):
        and_count = 0 
        kl_count = 0
        wd_count = 0
        ks_count = 0 
        for j in range(len(original_segments)):

            if kl_test(detected_segments[i], original_segments[j],threshold=0.2):
                if i ==j:
                    kl_count = kl_count+1

            if anderson_test(detected_segments[i], original_segments[j], alpha=0.05):
                if i ==j:
                    and_count = and_count +1

            if ks_test(detected_segments[i], original_segments[j], alpha=0.05):
                if i ==j:
                    ks_count = ks_count+1

            if wd_test(detected_segments[i], original_segments[j]):
                if i ==j:
                    wd_count = wd_count+1

        result_dict[(i,"Anderson")] = and_count
        result_dict[(i,"Kolmogorov")] = ks_count
        result_dict[(i,"wasserstein")] = ks_count
        result_dict[(i,"Kullback-Leibler")] = kl_count

    detected_seg_num = [key[0] for key in result_dict.keys()]
    test_name = [key[1] for key in result_dict.keys()]
    test_value = list(result_dict.values())
    result = pd.DataFrame({'detected_seg_num': detected_seg_num , 'test_name': test_name, 'test_value': test_value})

   
    labels={}
    for test_name in set(result['test_name']):
        df = result[result['test_name'] == test_name]
        zero_indices = [index for index, value in enumerate(df["test_value"]) if value == 0]
        updated_list = [value for index, value in enumerate(predicted_breaks) if index not in zero_indices]
        list1 =  np.zeros(n_samples)

        for bp in updated_list:
            if bp < n_samples:
                list1[bp] = 1
        
        
        # Calculate true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
        TP = np.sum((true_labels == 1) & (list1 == 1))
        FP = np.sum((true_labels == 0) & (list1 == 1))
        TN = np.sum((true_labels == 0) & (list1 == 0))
        FN = np.sum((true_labels == 1) & (list1 == 0))

        # Calculate recall and precision
        labels[f"recall_{test_name}"]  = TP / (TP + FN)
        labels[f"precision_{test_name}"]  = TP / (TP + FP)

        # Calculate false positive rate (FPR)
        fpr, _, _ = roc_curve(true_labels, list1)

        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, list1)

        # Calculate area under the ROC curve (AUC-ROC)
        auc_roc = auc(fpr, recall_curve)

        # Plot the ROC curve
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, recall_curve, marker='.')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('Recall')
        plt.title('ROC Curve (AUC = {:.3f})'.format(auc_roc))

        # Plot the precision-recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"Precision_Recall_Curve_{test_name}")
        plt.tight_layout()
        plt.show()
    
    return labels



def detection_based_roc(true_labels,predicted_labels):

    """
    This is based on rapture module detection
    """
    precision=[]
    recall = []
    true_positive = np.sum((predicted_labels == 1) & (true_labels == 1))
    false_negative = np.sum((predicted_labels == 0) & (true_labels == 1))
    false_positive = np.sum((predicted_labels == 1) & (true_labels == 0))
    recall.append(true_positive / (true_positive + false_negative))
    precision.append(true_positive / (true_positive + false_positive))

    # Calculate false positive rate (FPR)
    fpr, _, _ = roc_curve(true_labels, predicted_labels)

    # Calculate precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, predicted_labels)

    # Calculate area under the ROC curve (AUC-ROC)
    auc_roc = auc(fpr, recall_curve)

    # Plot the ROC curve
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, recall_curve, marker='.')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('Recall')
    plt.title('ROC Curve (AUC = {:.3f})'.format(auc_roc))

    # Plot the precision-recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()

    plt.show()
    return precision,recall
      

       
# Execute the main function
if __name__ == "__main__":
    data, original_segments, true_labels = simulate_data(n_samples = 1000,n_breaks = 50,pre_break_mean=0.5, post_break_mean=0.8)
    predicted_labels, detected_segments, predicted_breaks  = detect_breaks(data,n_bkps=40,n_samples=1000)
    precision_recall_dict = test_based_labels(detected_segments,original_segments,predicted_breaks,predicted_labels,true_labels,n_samples=1000)

    rapture_precision, rapture_recall = detection_based_roc(true_labels,predicted_labels)

          








































































































