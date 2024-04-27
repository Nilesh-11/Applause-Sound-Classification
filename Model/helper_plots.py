import matplotlib.pyplot as plt

# Data
accuracy = [0.9191, 0.9612, 0.9651, 0.9733, 0.9784, 0.9832, 0.9883, 0.9799, 0.9907, 0.9958, 0.9981, 0.9973, 0.9934]
loss = [0.7171, 0.1153, 0.0994, 0.0848, 0.0698, 0.0491, 0.0355, 0.0713, 0.0330, 0.0154, 0.0094, 0.0092, 0.0320]
val_accuracy = [0.9523, 0.9666, 0.9666, 0.9707, 0.9651, 0.9738, 0.9630, 0.9697, 0.9733, 0.9790, 0.9687, 0.9697, 0.9687]
val_loss = [0.1571, 0.1271, 0.0922, 0.1001, 0.1007, 0.0828, 0.1313, 0.1212, 0.1039, 0.0988, 0.1401, 0.1497, 0.1924]

epochs = range(1, len(accuracy) + 1)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("./Train_results.jpg")
plt.show()
