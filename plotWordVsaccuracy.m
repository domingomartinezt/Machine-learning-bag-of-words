word = [100, 200, 300, 400, 500];
accuracy_train = [89.99, 94.52, 96.18, 94.36, 95.30];
accuracy_test = [86.16, 93.04, 91.52, 91.02, 90.54];
figure
plot(word, accuracy_train, word, accuracy_test, 'LineWidth', 2)
ylabel('Accuracy');              
xlabel('Words');                
legend('Train','Test', 'Location','northwest')
title('Accuracy vs Number of Features')