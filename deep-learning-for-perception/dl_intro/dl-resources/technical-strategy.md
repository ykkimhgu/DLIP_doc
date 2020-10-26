# Technical Strategy by A.Ng

### Machine Learning Yearning - Andrew Ng

This is highlights of important technical strategy from the book.  You should read the book for more detailed explanation. To get the e-book [click here](https://www.deeplearning.ai/machine-learning-yearning/)

{% file src="../../../.gitbook/assets/ng-mly01-13.pdf" caption="Machine Learning Yearning - Andrew Ng" %}

### Setting up development and test sets

* Choose dev and test sets from a distribution that reflects what data you expect to get in the future and want to do well on. This may not be the same as your training data’s distribution.
*  Choose dev and test sets from the same distribution if possible.
*  Choose a single-number evaluation metric for your team to optimize. If there are multiple goals that you care about, consider combining them into a single formula \(such as averaging multiple error metrics\) or defining satisficing and optimizing metrics.
* Machine learning is a highly iterative process: You may try many dozens of ideas before finding one that you’re satisfied with.
* Having dev/test sets and a single-number evaluation metric helps you quickly evaluate algorithms, and therefore iterate faster.
* When starting out on a brand new application, try to establish dev/test sets and a metric quickly, say in less than a week. It might be okay to take longer on mature applications.
*  The old heuristic of a 70%/30% train/test split does not apply for problems where you have a lot of data; the dev and test sets can be much less than 30% of the data.
*  Your dev set should be large enough to detect meaningful changes in the accuracy of your algorithm, but not necessarily much larger. Your test set should be big enough to give you a confident estimate of the final performance of your system.
* If your dev set and metric are no longer pointing your team in the right direction, quickly change them: \(i\) If you had overfit the dev set, get more dev set data. \(ii\) If the actual distribution you care about is different from the dev/test set distribution, get new dev/test set data. \(iii\) If your metric is no longer measuring what is most important to you, change the metric.

### Basic error analysis

*  When you start a new project, especially if it is in an area in which you are not an expert, it is hard to correctly guess the most promising directions. 
* So don’t start off trying to design and build the perfect system. Instead build and train a basic system as quickly as possible—perhaps in a few days. Then use error analysis to help you identify the most promising directions and iteratively improve your algorithm from there. 
* Carry out error analysis by manually examining ~100 dev set examples the algorithm misclassifies and counting the major categories of errors. Use this information to prioritize what types of errors to work on fixing. 
* Consider splitting the dev set into an Eyeball dev set, which you will manually examine, and a Blackbox dev set, which you will not manually examine. If performance on the Eyeball dev set is much better than the Blackbox dev set, you have overfit the Eyeball dev set and should consider acquiring more data for it. 
* The Eyeball dev set should be big enough so that your algorithm misclassifies enough examples for you to analyze. A Blackbox dev set of 1,000-10,000 examples is sufficient for many applications. 
* If your dev set is not big enough to split this way, just use the entire dev set as an Eyeball dev set for manual error analysis, model selection, and hyperparameter tuning.

### Bias vs. Variance tradeoff

**Bias**: higher error rate than optimal accuracy on the **training set**. \(_underfitting_\) --&gt; more complex model

**Variance**: how much worse the algorithm does on the **dev \(or test\) set** than the training set. \(_overfitting_\) --&gt; add data to training

> Bias:  difference of \(TrainError-OptimalError\), avoidable bias
>
> Variance:  difference of  \(TestError - TrainingError\)

 **Tradeoff** 

* For example, increasing the size of your model—adding neurons/layers in a neural network, or adding input features—generally reduces bias but could increase variance. Alternatively, adding regularization generally increases bias but reduces variance.
* In the modern era, we often have access to plentiful data and can use very large neural networks \(deep learning\). Therefore, there is less of a tradeoff, and there are now more options for reducing bias without hurting variance, and vice versa.

### Techniques for reducing avoidable bias

If your learning algorithm suffers from high avoidable bias, you might try the following techniques: 

*  **Increase the model size** \(such as number of neurons/layers\): This technique reduces bias, since it should allow you to fit the training set better. If you find that this increases variance, then use regularization, which will usually eliminate the increase in variance. 
* **Modify input features based on insights from error analysis** : Say your error analysis inspires you to create additional features that help the algorithm eliminate a particular category of errors. \(We discuss this further in the next chapter.\) These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance. 
* **Reduce or eliminate regularization** \(L2 regularization, L1 regularization, dropout\): This will reduce avoidable bias, but increase variance. 
* **Modify model architecture** \(such as neural network architecture\) so that it is more suitable for your problem: This technique can affect both bias and variance. 

One method that is NOT helpful: 

*  A**dd more training data** : This technique helps with variance problems, but it usually has no significant effect on bias.

### Techniques for reducing variance

If your learning algorithm suffers from high variance, you might try the following techniques: 

* Add more training data : This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data. 
* Add regularization \(L2 regularization, L1 regularization, dropout\): This technique reduces variance but increases bias. 
* Add early stopping \(i.e., stop gradient descent early, based on dev set error\): This technique reduces variance but increases bias. Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique. 
* Feature selection to decrease number/type of input features: This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly \(say going from 1,000 features to 900\) is unlikely to have a huge effect on bias. Reducing it significantly \(say going from 1,000 features to 100—a 10x reduction\) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful. 
* Decrease the model size \(such as number of neurons/layers\): Use with caution. This technique could decrease variance, while possibly increasing bias. However, I don’t recommend this technique for addressing variance. Adding regularization usually gives better classification performance. The advantage of reducing the model size is reducing your computational cost and thus speeding up how quickly you can train models. If speeding up model training is useful, then by all means consider decreasing the model size. But if your goal is to reduce variance, and you are not concerned about the computational cost, consider adding regularization instead.

Common tactics for both bias and variance:

* **Modify input features based on insights from error analysis :** Say your error analysis inspires you to create additional features that help the algorithm to eliminate a particular category of errors. These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.
* **Modify model architecture** \(such as neural network architecture\) so that it is more suitable for your problem: This technique can affect both bias and variance.

### Plotting learning curves

**Underfit curve**

* The training loss remains flat regardless of training.
* The training loss continues to decrease until the end of training
* The training loss is much higher than the optimal performance

![Image source from \(machinelearningmastery.com\)](../../../.gitbook/assets/image%20%283%29.png)

**Overfit curve**

* The plot of training loss continues to decrease with experience.
* The plot of validation loss decreases to a point and begins increasing again.
* The plot of validation loss is much higher than training loss

![Image source from \(machinelearningmastery.com\)](../../../.gitbook/assets/image%20%287%29.png)

**Other examples**

* Low Variance and High Bias

![](../../../.gitbook/assets/image%20%286%29.png)

* High Variance, and Low bias --&gt; add more training data

![](../../../.gitbook/assets/image%20%284%29.png)

* High Variance and High Bias

![](../../../.gitbook/assets/image%20%282%29.png)

#### For very small training sets

The learning curve with small training sets could be noisy. If the noise in the training curve makes it hard to see the true trends, here are solutions:\`

* This may occur if the training dataset has _too few examples as compared to the validation dataset_.
* Instead of training just one model on 10 examples, select several \(say 3-10\) different randomly chosen training sets of 10 examples by sampling with replacement10 from your original set of 100. Train a different model on each of these, and compute the training and dev set error of each of the resulting models. Compute and plot the average training error and average dev set error.
* If your training set is skewed towards one class, or if it has many classes, choose a “balanced” subset instead of 10 training examples at random out of the set of 100. For example, you can make sure that 2/10 of the examples are positive examples, and 8/10 are negative. More generally, you can make sure the fraction of examples from each class is as close as possible to the overall fraction in the original training set.

### Further Reading

#### Working with small data sets

{% embed url="https://towardsdatascience.com/breaking-the-curse-of-small-data-sets-in-machine-learning-part-2-894aa45277f4" %}

{% embed url="https://towardsdatascience.com/how-to-use-deep-learning-even-with-small-data-e7f34b673987" %}





