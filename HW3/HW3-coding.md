{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# HW3-Coding\n",
                "\n",
                "## Q4 Kernel Power on SVM and Regularized Logistic Regression\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import plot_roc_curve, roc_auc_score, f1_score, accuracy_score\n",
                "\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_Q4 = pd.read_csv('nonlineardata.csv', header=None)\n",
                "data_Q4.columns=['x1','x2', 'label']\n",
                "data_Q4.head()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "<div>\n",
                "\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "<style scoped>\n",
                "    .dataframe tbody tr th:only-of-type {\n",
                "        vertical-align: middle;\n",
                "    }\n",
                "\n",
                "\n",
                "<\/style>\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "+---+-----------+-----------+-------+\n",
                "|   | x1        | x2        | label |\n",
                "+===+===========+===========+=======+\n",
                "| 0 | 0.107878  | 0.132034  | 1.0   |\n",
                "+---+-----------+-----------+-------+\n",
                "| 1 | 0.728726  | 0.287441  | 1.0   |\n",
                "+---+-----------+-----------+-------+\n",
                "| 2 | -0.130377 | -0.483445 | -1.0  |\n",
                "+---+-----------+-----------+-------+\n",
                "| 3 | 0.009136  | -0.465431 | -1.0  |\n",
                "+---+-----------+-----------+-------+\n",
                "| 4 | 0.790434  | 0.192522  | 1.0   |\n",
                "+---+-----------+-----------+-------+\n",
                "\n",
                "<\/div>\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# plot the data points\n",
                "\n",
                "plt.figure(figsize=(8,6))\n",
                "pos_idx = data_Q4['label'] == 1.0\n",
                "plt.scatter(data_Q4['x1'].loc[pos_idx], data_Q4['x2'].loc[pos_idx], \n",
                "            color='red', alpha=0.6)\n",
                "neg_idx = data_Q4['label'] == -1.0\n",
                "plt.scatter(data_Q4['x1'].loc[neg_idx], data_Q4['x2'].loc[neg_idx], \n",
                "            color='blue', alpha=0.6);\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_3_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# split data\n",
                "\n",
                "X_q4 = data_Q4[['x1','x2']]\n",
                "y_q4 = data_Q4['label']\n",
                "X_train, X_test, y_train, y_test = train_test_split(X_q4, y_q4, \n",
                "                                                    test_size=0.20, random_state=2022)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plotClassifier(model, X, y):\n",
                "    \"\"\"plots the decision boundary of the model and the scatterpoints\n",
                "       of the target values 'y'.\n",
                "\n",
                "    Assumptions\n",
                "    -----------\n",
                "    y : it should contain two classes: '1' and '2'\n",
                "\n",
                "    Parameters\n",
                "    ----------\n",
                "    model : the trained model which has the predict function\n",
                "\n",
                "    X : the N by D feature array\n",
                "\n",
                "    y : the N element vector corresponding to the target values\n",
                "\n",
                "    \"\"\"\n",
                "    x1 = X[:, 0]\n",
                "    x2 = X[:, 1]\n",
                "\n",
                "    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1\n",
                "    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1\n",
                "\n",
                "    x1_line =  np.linspace(x1_min, x1_max, 200)\n",
                "    x2_line =  np.linspace(x2_min, x2_max, 200)\n",
                "\n",
                "    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)\n",
                "\n",
                "    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]\n",
                "\n",
                "    y_pred = np.array(model.predict(mesh_data))\n",
                "    y_pred = np.reshape(y_pred, x1_mesh.shape)\n",
                "\n",
                "    plt.figure()\n",
                "    plt.xlim([x1_mesh.min(), x1_mesh.max()])\n",
                "    plt.ylim([x2_mesh.min(), x2_mesh.max()])\n",
                "\n",
                "    plt.contourf(x1_mesh, x2_mesh, -y_pred.astype(int), # unsigned int causes problems with negative sign... o_O\n",
                "                cmap=plt.cm.RdBu, alpha=0.6)\n",
                "\n",
                "\n",
                "    y_vals = np.unique(y)\n",
                "    plt.scatter(x1[y==y_vals[0]], x2[y==y_vals[0]], color=\"b\", label=\"class %+d\" % y_vals[0])\n",
                "    plt.scatter(x1[y==y_vals[1]], x2[y==y_vals[1]], color=\"r\", label=\"class %+d\" % y_vals[1])\n",
                "    plt.legend()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### SVC with different kernels\n",
                "\n",
                "### 4.1. Train a soft-margin SVM with linear kernel and C = 100\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.gridspec as gridspec\n",
                "from mlxtend.plotting import plot_decision_regions\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "C1=100\n",
                "# fit the svc model\n",
                "svc_clf_l = SVC(kernel='linear', C=C1)\n",
                "svc_clf_l.fit(X_q4, y_q4)\n",
                "\n",
                "fig = plt.subplots(figsize=(8,6))\n",
                "plot_decision_regions(np.array(X_train), np.array(y_train.astype(int)), svc_clf_l);\n",
                "plotClassifier(svc_clf_l, np.array(X_train), np.array(y_train));\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_8_0.png) ​\n",
                "\n",
                "![png](output_8_1.png)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_pred = svc_clf_l.predict(X_test)\n",
                "y_train_pred = svc_clf_l.predict(X_train)\n",
                "training_error_l = 1 - accuracy_score(y_train, y_train_pred)\n",
                "test_error_l = 1 - accuracy_score(y_test, y_pred)\n",
                "print('training error  when using linear kernel is:')\n",
                "print(training_error_l)\n",
                "\n",
                "print('testing error  when using linear kernel is:')\n",
                "print(test_error_l)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    training error  when using linear kernel is:\n",
                "    0.19687500000000002\n",
                "    testing error  when using linear kernel is:\n",
                "    0.23750000000000004\n",
                "\n",
                "### 4.2. Train a soft-margin SVM with polynomial kernel\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# fit the svc model\n",
                "svc_clf_poly = SVC(kernel='poly', C=C1)\n",
                "svc_clf_poly.fit(X_q4, y_q4)\n",
                "\n",
                "fig = plt.subplots(figsize=(8,6))\n",
                "plot_decision_regions(np.array(X_train), np.array(y_train.astype(int)), svc_clf_poly);\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_11_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_pred = svc_clf_poly.predict(X_test)\n",
                "y_train_pred_p = svc_clf_poly.predict(X_train)\n",
                "training_error_poly = 1 - accuracy_score(y_train, y_train_pred_p)\n",
                "test_error_poly = 1 - accuracy_score(y_test, y_pred)\n",
                "print('training error  when using polynomial kernel is:')\n",
                "print(training_error_poly)\n",
                "\n",
                "print('testing error  when using polynomial kernel is:')\n",
                "print(test_error_poly)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    training error  when using linear kernel is:\n",
                "    0.21562499999999996\n",
                "    testing error  when using linear kernel is:\n",
                "    0.26249999999999996\n",
                "\n",
                "### 4.3. Train a soft-margin SVM with RBF kernel\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# fit the svc model\n",
                "svc_clf_rbf = SVC(kernel='rbf', C=C1)\n",
                "svc_clf_rbf.fit(X_q4, y_q4)\n",
                "\n",
                "fig = plt.subplots(figsize=(8,6))\n",
                "plot_decision_regions(np.array(X_train), np.array(y_train.astype(int)), svc_clf_rbf);\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_14_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_pred = svc_clf_rbf.predict(X_test)\n",
                "y_train_pred_rbf = svc_clf_rbf.predict(X_train)\n",
                "training_error_rbf = 1 - accuracy_score(y_train, y_train_pred_rbf)\n",
                "test_error_rbf = 1 - accuracy_score(y_test, y_pred)\n",
                "print('training error  when using rbf kernel is:')\n",
                "print(training_error_rbf)\n",
                "\n",
                "print('testing error  when using rbf kernel is:')\n",
                "print(test_error_rbf)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    training error  when using rbf kernel is:\n",
                "    0.10312500000000002\n",
                "    testing error  when using rbf kernel is:\n",
                "    0.15000000000000002\n",
                "\n",
                "### 4.4. RBF with different gammas\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]\n",
                "\n",
                "log_gamma = [np.log(g) for g in gammas]\n",
                "\n",
                "# def PlotRBFError():\n",
                "train_error_rbf = []\n",
                "test_error_rbf = []\n",
                "# train SVC with different gammas\n",
                "for gamma in gammas:\n",
                "    \n",
                "    svc_rbf = SVC(kernel='rbf', C=C1, gamma = gamma)\n",
                "    svc_rbf.fit(X_train, y_train)\n",
                "    # report training error\n",
                "    y_train_pred = svc_rbf.predict(X_train)\n",
                "    train_error = 1 - accuracy_score(y_train, y_train_pred)\n",
                "    train_error_rbf.append(train_error)\n",
                "    \n",
                "    # report testing error\n",
                "    y_pred = svc_rbf.predict(X_test)\n",
                "    test_error = 1 - accuracy_score(y_test, y_pred)\n",
                "    test_error_rbf.append(test_error)\n",
                "    \n",
                "\n",
                "    \n",
                "# plot train and test error w.r.t. gamma\n",
                "fig = plt.subplots(figsize=(8,6))\n",
                "plt.plot(log_gamma, train_error_rbf, color='blue', alpha=0.6)\n",
                "plt.plot(log_gamma, test_error_rbf, color='red', alpha=0.6)\n",
                "plt.title(u'training and testing error vs. $\\gamma$');\n",
                "plt.legend(['train','test'])\n",
                "plt.xlabel(u'log$\\gamma$')\n",
                "plt.ylabel('error rate');\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_17_0.png) ​\n",
                "\n",
                "What do you notice about the error rates as $\\gamma$ increases? Why this\n",
                "is the case? Finally, in the bias variance tradeoff, does small $\\gamma$\n",
                "correspond to high bias or high variance?\\\n",
                "Answer:\n",
                "\n",
                "I used F1 score to show error rate. As $\\gamma$ gets higher, f1 score\n",
                "increases, which means error rate gets lower.\n",
                "\n",
                "As $\\gamma$ increases, the decision boundary will be closer to data\n",
                "points, and thus could classify more accurately. However, if $\\gamma$\n",
                "gets too high, it may overfit.\n",
                "\n",
                "Small $\\gamma$ correspond to high bias. Because it tends to underfit.\n",
                "\n",
                "### Logistic Regression with Different Kernels\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pickle\n",
                "import os\n",
                "from sklearn.model_selection import train_test_split\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from scipy.optimize import approx_fprime\n",
                "from numpy.linalg import norm\n",
                "import math\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "def check_gradient(model, X, y, dimensionality, verbose=True):\n",
                "    # This checks that the gradient implementation is correct\n",
                "    w = np.random.rand(dimensionality)\n",
                "    f, g = model.funObj(w, X, y)\n",
                "\n",
                "    # Check the gradient\n",
                "    estimated_gradient = approx_fprime(w,\n",
                "                                       lambda w: model.funObj(w,X,y)[0],\n",
                "                                       epsilon=1e-6)\n",
                "\n",
                "    implemented_gradient = model.funObj(w, X, y)[1]\n",
                "\n",
                "    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-3):\n",
                "        raise Exception('User and numerical derivatives differ:\\n%s\\n%s' %\n",
                "             (estimated_gradient[:5], implemented_gradient[:5]))\n",
                "    else:\n",
                "        if verbose:\n",
                "            print('User and numerical derivatives agree.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "def log_1_plus_exp_safe(x):\n",
                "    out = np.log(1+np.exp(x))\n",
                "    out[x > 100] = x[x>100]\n",
                "    out[x < -100] = np.exp(x[x < -100])\n",
                "    return out\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "def findMin(funObj, w, maxEvals, *args, verbose=0):\n",
                "    \"\"\"\n",
                "    Uses gradient descent to optimize the objective function\n",
                "\n",
                "    This uses quadratic interpolation in its line search to\n",
                "    determine the step size alpha\n",
                "    \"\"\"\n",
                "    # Parameters of the Optimization\n",
                "    optTol = 1e-2\n",
                "    gamma = 1e-4\n",
                "\n",
                "    # Evaluate the initial function value and gradient\n",
                "    f, g = funObj(w,*args)\n",
                "    funEvals = 1\n",
                "\n",
                "    alpha = 1.\n",
                "    while True:\n",
                "        # Line-search using quadratic interpolation to \n",
                "        # find an acceptable value of alpha\n",
                "        gg = g.T.dot(g)\n",
                "\n",
                "        while True:\n",
                "            w_new = w - alpha * g\n",
                "            f_new, g_new = funObj(w_new, *args)\n",
                "\n",
                "            funEvals += 1\n",
                "            if f_new <= f - gamma * alpha*gg:\n",
                "                break\n",
                "\n",
                "            if verbose > 1:\n",
                "                print(\"f_new: %.3f - f: %.3f - Backtracking...\" % (f_new, f))\n",
                "\n",
                "            # Update step size alpha\n",
                "            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))\n",
                "\n",
                "        # Print progress\n",
                "        if verbose > 0:\n",
                "            print(\"%d - loss: %.3f\" % (funEvals, f_new))\n",
                "\n",
                "        # Update step-size for next iteration\n",
                "        y = g_new - g\n",
                "        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)\n",
                "\n",
                "        # Safety guards\n",
                "        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:\n",
                "            alpha = 1.\n",
                "\n",
                "        if verbose > 1:\n",
                "            print(\"alpha: %.3f\" % (alpha))\n",
                "\n",
                "        # Update parameters/function/gradient\n",
                "        w = w_new\n",
                "        f = f_new\n",
                "        g = g_new\n",
                "\n",
                "        # Test termination conditions\n",
                "        optCond = norm(g, float('inf'))\n",
                "\n",
                "        if optCond < optTol:\n",
                "            if verbose:\n",
                "                print(\"Problem solved up to optimality tolerance %.3f\" % optTol)\n",
                "            break\n",
                "\n",
                "        if funEvals >= maxEvals:\n",
                "            if verbose:\n",
                "                print(\"Reached maximum number of function evaluations %d\" % maxEvals)\n",
                "            break\n",
                "\n",
                "    return w, f\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "class kernelLogRegL2():\n",
                "    def __init__(self, lammy=1.0, verbose=0, maxEvals=100, \n",
                "                 kernel_fun=kernel_linear, **kernel_args):\n",
                "        self.verbose = verbose\n",
                "        self.lammy = lammy\n",
                "        self.maxEvals = maxEvals\n",
                "        self.kernel_fun = kernel_fun\n",
                "        self.kernel_args = kernel_args\n",
                "\n",
                "    def funObj(self, u, K, y):\n",
                "        yKu = y * (K@u)\n",
                "\n",
                "        # Calculate the function value\n",
                "        # f = np.sum(np.log(1. + np.exp(-yKu)))\n",
                "        f = np.sum(log_1_plus_exp_safe(-yKu))\n",
                "\n",
                "        # Add L2 regularization\n",
                "        f += 0.5 * self.lammy * u.T@K@u\n",
                "\n",
                "        # Calculate the gradient value\n",
                "        res = - y / (1. + np.exp(yKu))\n",
                "        g = (K.T@res) + self.lammy * K@u\n",
                "\n",
                "        return f, g\n",
                "\n",
                "\n",
                "    def fit(self, X, y):\n",
                "        n, d = X.shape\n",
                "        self.X = X\n",
                "\n",
                "        K = self.kernel_fun(X,X, **self.kernel_args)\n",
                "\n",
                "        check_gradient(self, K, y, n, verbose=self.verbose)\n",
                "        self.u, f = findMin(self.funObj, np.zeros(n), self.maxEvals, \n",
                "                            K, y, verbose=self.verbose)\n",
                "\n",
                "    def predict(self, Xtest):\n",
                "        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)\n",
                "        return np.sign(Ktest@self.u)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "def kernel_linear(X1, X2):\n",
                "    return X1 @ X2.T\n",
                "\n",
                "def kernel_poly(X1, X2, p=2):\n",
                "    #Your code here\n",
                "    return (1 + X1 @ X2.T)**2\n",
                "\n",
                "\n",
                "def kernel_RBF(X1, X2, sigma=0.5):\n",
                "    \n",
                "    mat = np.array([[1]* len(X2) for x in range(len(X1))])\n",
                "        \n",
                "    for i in range(len(X1)):\n",
                "        for j in range(len(X2)):\n",
                "            mat[i][j] = math.dist(X1[i,:], X2[j,:])\n",
                "\n",
                "    result = np.exp(-1 * mat / sigma**2)\n",
                "\n",
                "    return result\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4.5. Train a L2 regularized Logistic Regression classifier with linear kernel, $\\lambda$ = 0.01, and report training and testing error.\n",
                "\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "kernelLog_linear = kernelLogRegL2(lammy=0.01)\n",
                "kernelLog_linear.fit(X_train, y_train)\n",
                "plotClassifier(kernelLog_linear, np.array(X_train), np.array(y_train))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_27_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_train_linear = kernelLog_linear.predict(X_train)\n",
                "y_pred_linear = kernelLog_linear.predict(X_test)\n",
                "log_l_train_error = 1 - accuracy_score(y_train, y_train_linear)\n",
                "log_l_test_error = 1 - accuracy_score(y_test, y_pred_linear)\n",
                "print('logistic regression training error when using linear kernel is:')\n",
                "print(log_l_train_error)\n",
                "\n",
                "print('logistic regression testing error when using linear kernel is:')\n",
                "print(log_l_test_error)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    logistic regression training error when using linear kernel is:\n",
                "    0.23750000000000004\n",
                "    logistic regression testing error when using linear kernel is:\n",
                "    0.23750000000000004\n",
                "\n",
                "### 4.6. Train a L2 regularized Logistic Regression classifier with polynomial kernel, $\\lambda$ = 0.01, and report training and testing error.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "kernelLog_poly = kernelLogRegL2(lammy=0.01, kernel_fun = kernel_poly)\n",
                "kernelLog_poly.fit(X_train, y_train)\n",
                "plotClassifier(kernelLog_poly, np.array(X_train), np.array(y_train))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_30_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_train_poly = kernelLog_poly.predict(X_train)\n",
                "y_pred_poly = kernelLog_poly.predict(X_test)\n",
                "log_p_train_error = 1 - accuracy_score(y_train, y_train_poly)\n",
                "log_p_test_error = 1 - accuracy_score(y_test, y_pred_poly)\n",
                "print('logistic regression training error when using polynomial kernel is:')\n",
                "print(log_p_train_error)\n",
                "\n",
                "print('logistic regression testing error when using polynomial kernel is:')\n",
                "print(log_p_test_error)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    logistic regression training error when using polynomial kernel is:\n",
                "    0.18437499999999996\n",
                "    logistic regression testing error when using polynomial kernel is:\n",
                "    0.23750000000000004\n",
                "\n",
                "### 4.7. Train a L2 regularized Logistic Regression classifier with RBF kernel, $\\lambda$ = 0.01, $\\sigma=0.5$, and report training and testing error.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "kernelLog_rbf = kernelLogRegL2(lammy=0.01, kernel_fun = kernel_RBF)\n",
                "kernelLog_rbf.fit(X_train.values, y_train.values)\n",
                "plotClassifier(kernelLog_rbf, np.array(X_train), np.array(y_train))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_33_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_train_rbf = kernelLog_rbf.predict(X_train.values)\n",
                "y_pred_rbf = kernelLog_rbf.predict(X_test.values)\n",
                "log_rbf_train_error = 1 - accuracy_score(y_train, y_train_rbf)\n",
                "log_rbf_test_error = 1 - accuracy_score(y_test, y_pred_rbf)\n",
                "print('logistic regression training error when using RBF kernel is:')\n",
                "print(log_rbf_train_error)\n",
                "\n",
                "print('logistic regression testing error when using RBF kernel is:')\n",
                "print(log_rbf_test_error)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    logistic regression training error when using RBF kernel is:\n",
                "    0.043749999999999956\n",
                "    logistic regression testing error when using RBF kernel is:\n",
                "    0.16249999999999998\n",
                "\n",
                "### 4.8. Try different $\\lambda$ and plot the training and test error rates vs $\\gamma$\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_error48 = []\n",
                "test_error48 = []\n",
                "gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]\n",
                "for gamma in gammas:\n",
                "    \n",
                "    sigma = np.sqrt(1 / gamma)\n",
                "    kernelLog_rbf48 = kernelLogRegL2(lammy=0.01, kernel_fun = kernel_RBF, sigma=sigma)\n",
                "    kernelLog_rbf48.fit(X_train.values, y_train.values)\n",
                "    y_train_rbf48 = kernelLog_rbf48.predict(X_train.values)\n",
                "    y_pred_rbf48 = kernelLog_rbf48.predict(X_test.values)\n",
                "    log_rbf_train_error48 = 1 - accuracy_score(y_train, y_train_rbf48)\n",
                "    log_rbf_test_error48 = 1 - accuracy_score(y_test, y_pred_rbf48)\n",
                "    train_error48.append(log_rbf_train_error48)\n",
                "    test_error48.append(log_rbf_test_error48)\n",
                "    \n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "log_gamma = [np.log(g) for g in gammas]\n",
                "fig48, ax48 = plt.subplots(figsize=(8,6))\n",
                "plt.plot(log_gamma, train_error48)\n",
                "plt.plot(log_gamma, test_error48);\n",
                "plt.xlabel(u'$\\gamma$')\n",
                "plt.ylabel('f1 score')\n",
                "plt.legend(['train', 'test'])\n",
                "plt.title(u'Training and test error vs $\\gamma$');\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_37_0.png) ​\n",
                "\n",
                "## Q5 Sparse Logistic Regression\n",
                "\n",
                "### 5.1 plot ROC curve\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn import metrics, model_selection\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_q5 = pd.read_csv('spectf.csv',header=None)\n",
                "df_q5.shape\n",
                "# df_q5.head()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    (267, 45)\n",
                "\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_q5 = df_q5.iloc[:,1:]\n",
                "y_q5 = df_q5.iloc[:, 0]\n",
                "X_train_q5, X_test_q5, y_train_q5, y_test_q5 = train_test_split(X_q5, y_q5, \n",
                "                                                    test_size=0.25, random_state=2022)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "features = X_train_q5.columns\n",
                "fig5, ax5 = plt.subplots(figsize=(8,6))\n",
                "# y_train5 = np.array(y_train_q5).reshape(-1,1)\n",
                "# y_test5 = np.array(y_test_q5).reshape(-1,1)\n",
                "# fig5, ax5 = plt.subplots(figsize=(8,6))\n",
                "aucs = {}\n",
                "for f in features:\n",
                "    X = np.array(X_train_q5[f]).reshape(-1,1)\n",
                "    log_clf = LogisticRegression(penalty='none',random_state=0).fit(X=X, y=y_train_q5)\n",
                "#     X_test5 = np.array(X_test_q5[f]).reshape(-1,1)\n",
                "    y_pred5 = log_clf.predict(X)\n",
                "    \n",
                "    metrics.plot_roc_curve(log_clf, X, y_train_q5, ax=ax5)\n",
                "    auc = metrics.roc_auc_score(y_train_q5, y_pred5)\n",
                "    aucs[f] = auc\n",
                "    ax5.get_legend().remove()\n",
                "# plt.show()\n",
                "    \n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_42_0.png) ​\n",
                "\n",
                "### 5.2 Create an algorithm for choosing 300 different subsets of these features\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Covariance matrix\n",
                "X_array = np.array(X_train_q5)\n",
                "cov = np.cov(X_array.T)\n",
                "mean_var = np.mean(cov)\n",
                "\n",
                "# extract features that have AUC score above 0.5\n",
                "auc_above = {}     # featurs with AUC above 0.5\n",
                "auc_low = {}      # all other features\n",
                "for key, value in aucs.items():\n",
                "    if value > 0.5:\n",
                "        auc_above[key] = value\n",
                "        \n",
                "    else:\n",
                "        auc_low[key] = value\n",
                "        \n",
                "\n",
                "# number of features\n",
                "n_subsets = 0\n",
                "subsets = []\n",
                "# for each such feature, iterate all other features\n",
                "while n_subsets < 300:\n",
                "    # for each feature with a higher AUC, keep it\n",
                "    # and search for other features\n",
                "    for k, v in auc_above.items():\n",
                "        for k1, v1 in aucs.items():\n",
                "            if k != k1:\n",
                "                cov_k = cov[k-1,k1-1]    # check the covriance\n",
                "                if cov_k < mean_var:     \n",
                "                    # only take two features that are not highly correlated\n",
                "                    subsets.append([k,k1])\n",
                "                    n_subsets += 1                    \n",
                "#     print(n_subsets)\n",
                "# while n_subsets < 300:\n",
                "    for k2, v2 in auc_low.items():\n",
                "        for k3, v3 in aucs.items():\n",
                "            if (k2 != k3) and ([k2,k3] or [k3,k2] not in subsets):\n",
                "                # exclude identical subsets\n",
                "                cov_k_ = cov[k2-1,k3-1]\n",
                "                if cov_k < mean_var:\n",
                "                    subsets.append([k2,k3])\n",
                "                    n_subsets += 1\n",
                "                    \n",
                "                    \n",
                "subsets1 = subsets[:300]                \n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "My algorithm:\n",
                "\n",
                "I search for subsets with two features. First divide the features into\n",
                "two groups, 'good 'features with higher AUC (above 0.5) and 'bad' ones\n",
                "with AUC of 0.5. Then for each subset, I want to keep one of these\n",
                "'good' features, and add another feature that is not highly correlated\n",
                "with it by checking the covariance. Repeat the steps until reaching 300\n",
                "subsets.\n",
                "\n",
                "### 5.3 Train 300 logistic regression models by calling a logistic regression solver on each subset of data\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "auc53 = {}\n",
                "fig53, ax53 = plt.subplots(figsize=(8,6))\n",
                "for idx, s in enumerate(subsets1):\n",
                "    X_3 = X_train_q5[[s[0],s[1]]]\n",
                "    log_clf53 = LogisticRegression(penalty='none',random_state=0).fit(X=X_3, y=y_train_q5)\n",
                "#     X_test5 = np.array(X_test_q5[f]).reshape(-1,1)\n",
                "    y_pred53 = log_clf53.predict(X_3)\n",
                "    # plot ROC\n",
                "    metrics.plot_roc_curve(log_clf53, X_3, y_train_q5, ax=ax53)\n",
                "    auc1 = metrics.roc_auc_score(y_train_q5, y_pred53)\n",
                "    auc53[idx] = auc1\n",
                "ax53.get_legend().remove()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_47_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "max_auc = max(auc53.values())\n",
                "# min(auc53.values())\n",
                "\n",
                "def ReturnKey(dictionary, value):\n",
                "    \n",
                "    for k,v in dictionary.items():\n",
                "        if v == value:\n",
                "            \n",
                "            key = k\n",
                "    return k\n",
                "\n",
                "idxmax = ReturnKey(auc53, max_auc)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "\n"
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# subset with the highest the AUC among all 300 subsets I chose\n",
                "subset_max = subsets[idxmax]\n",
                "subset_max\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "    [43, 36]\n",
                "\n",
                "### 5.4 Put the training ROC for f and the test ROC for f on the same figure and report the test AUC.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train_best = X_train_q5[[subset_max[0],subset_max[1]]]\n",
                "X_test_best = X_test_q5[[subset_max[0],subset_max[1]]]\n",
                "log_clf_best = LogisticRegression(penalty='none',random_state=0).fit(\n",
                "                                        X=X_train_best, y=y_train_q5)\n",
                "\n",
                "y_train_pred = log_clf_best.predict(X_train_best)\n",
                "y_test_pred = log_clf_best.predict(X_test_best)\n",
                "\n",
                "fig54, ax54 = plt.subplots(figsize=(8,6))\n",
                "metrics.plot_roc_curve(log_clf_best, X_train_best, y_train_q5, ax=ax54)\n",
                "metrics.plot_roc_curve(log_clf_best, X_test_best, y_test_q5, ax=ax54);\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_51_0.png) ​\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# l2 regularized logistic regression\n",
                "\n",
                "log_clf_l2 = LogisticRegression(penalty='l2',random_state=0).fit(\n",
                "                                        X=X_train_best, y=y_train_q5)\n",
                "\n",
                "y_train_pred = log_clf_l2.predict(X_train_best)\n",
                "y_test_pred = log_clf_l2.predict(X_test_best)\n",
                "\n",
                "fig541, ax541 = plt.subplots(figsize=(8,6))\n",
                "metrics.plot_roc_curve(log_clf_l2, X_train_best, y_train_q5, ax=ax541)\n",
                "metrics.plot_roc_curve(log_clf_l2, X_test_best, y_test_q5, ax=ax541);\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "​\\\n",
                "![png](output_52_0.png) ​\n",
                "\n",
                "It seems this technique works roughly the same as l2-regularized\n",
                "logistic regression.\n"
            ]
        }
    ],
    "metadata": {
        "anaconda-cloud": "",
        "kernelspec": {
            "display_name": "R",
            "langauge": "R",
            "name": "ir"
        },
        "language_info": {
            "codemirror_mode": "r",
            "file_extension": ".r",
            "mimetype": "text/x-r-source",
            "name": "R",
            "pygments_lexer": "r",
            "version": "3.4.1"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 1
}
