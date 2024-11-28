# 7. Compute Accuracy, Error rate, Precision, Recall for following confusion
# matrix ( Use formula for each)
# True Positives (TPs): 1 False Positives (FPs): 1
# False Negatives (FNs): 8 True Negatives (TNs): 90




#17

# Confusion Matrix values
TP = 1  # True Positives
FP = 1  # False Positives
FN = 8  # False Negatives
TN = 90 # True Negatives

# Calculations
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# Output the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")




























# 22. Compute Accuracy, Error rate, Precision, Recall for the following
# confusion matrix.
# Actual Class\Predicted
# class
# cancer =
# yes
# cancer = no Total
# cancer = yes 90 210 300
# cancer = no 140 9560 9700
# Total 230 9770 10000




#22

# Given values
TP = 90  # True Positives (cancer=yes, predicted=cancer=yes)
FP = 140  # False Positives (cancer=no, predicted=cancer=yes)
FN = 210  # False Negatives (cancer=yes, predicted=cancer=no)
TN = 9560  # True Negatives (cancer=no, predicted=cancer=no)

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Error Rate
error_rate = (FP + FN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP)

# Recall (Sensitivity)
recall = TP / (TP + FN)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

