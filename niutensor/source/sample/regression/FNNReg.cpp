#include "FNNReg.h"
#include "../../tensor/function/FHeader.h"
namespace fnnreg
{
/*base parameter*/
float learningRate = 0.3F;           // learning rate
int nEpoch = 200;                      // max training epochs
float minmax = 0.01F;                 // range [-p,p] for parameter initialization

void Init(FNNRegModel &model);
void InitGrad(FNNRegModel &model, FNNRegModel &grad);
void Train(float *trainDataX, float *trainDataY, int dataSize, FNNRegModel &model, int batchSize);
void Forword(XTensor &input, FNNRegModel &model, FNNRegNet &net);
void MSELoss(XTensor &output, XTensor &gold, XTensor &loss);
void Backward(XTensor &input, XTensor &gold, FNNRegModel &model, FNNRegModel &grad, FNNRegNet &net);
void Update(FNNRegModel &model, FNNRegModel &grad, float learningRate);
void CleanGrad(FNNRegModel &grad);
void Test(float *testData, int testDataSize, FNNRegModel &model);

int FNNRegMain(int argc, const char ** argv)
{
    FNNRegModel model;
	int batch_size = int(argv[1]);
    model.h_size = 4;
    const int dataSize = 16;
    const int testDataSize = 3;
    model.devID = -1;
	/*int i;
	for (i = 0; i < argc; i++)
		printf("Argument %d is %s.\n", i, argv[i]);*/
    Init(model);

    /*train Data*/
    float trainDataX[dataSize] = { 51, 56.8, 58,   63,   66,   69, 73,   76, 81, 85, 90,   94,  97,   100, 103,  107 };
    float trainDataY[dataSize] = { 31, 34.7, 35.6, 36.7, 39.5, 42, 42.7, 47, 49, 51, 52.5, 54,  55.7, 56,  58.8, 59.2 };

    float testDataX[testDataSize] = { 64, 80, 95 };

	
    Train(trainDataX, trainDataY, dataSize, model, 2);
	printf("train finished\n");
    Test(testDataX, testDataSize, model);
    return 0;
}

void Init(FNNRegModel &model)
{
	//weight1 1*4
    InitTensor2D(&model.weight1, 1, model.h_size, X_FLOAT, model.devID);
    //weight2 4*1
	InitTensor2D(&model.weight2, model.h_size, 1, X_FLOAT, model.devID);
    //b 1*4 todo
	InitTensor2D(&model.b, 2, model.h_size, X_FLOAT, model.devID);
    model.weight1.SetDataRand(-minmax, minmax);
    model.weight2.SetDataRand(-minmax, minmax);
    model.b.SetZeroAll();
    printf("Init model finish!\n");
}

void InitGrad(FNNRegModel &model, FNNRegModel &grad)
{
    InitTensor(&grad.weight1, &model.weight1);
    InitTensor(&grad.weight2, &model.weight2);
    InitTensor(&grad.b, &model.b);

    grad.h_size = model.h_size;
    grad.devID = model.devID;
}

void Train(float *trainDataX, float *trainDataY, int dataSize, FNNRegModel &model, int batchSize)
{
    printf("prepare data for train\n");
    /*prepare for train*/
    TensorList inputList;
    TensorList goldList;
    for (int i = 0; i < dataSize/batchSize; ++i)
    {
        XTensor*  inputData = NewTensor2D(2, 1, X_FLOAT, model.devID);
		XTensor* goldData = NewTensor2D(2, 1, X_FLOAT, model.devID);

		for (int j = 0; j < batchSize; ++j) {
			inputData->Set2D(trainDataX[i*batchSize+j] / 100, j, 0);
			goldData->Set2D(trainDataY[i * batchSize + j] / 60, j, 0);
			
		}
		
		inputList.Add(inputData);
		goldList.Add(goldData);

        
    }

    /*check data*/
    /*
    for (int i = 0; i < 16; ++i)
    {
        XTensor* tmp = inputList.GetItem(i);
        tmp->Dump(stderr);

        tmp = goldList.GetItem(i);
        tmp->Dump(stderr);
    }
    */

    printf("start train\n");
    FNNRegNet net;
    FNNRegModel grad;
    InitGrad(model, grad);
    for (int epochIndex = 0; epochIndex < nEpoch; ++epochIndex)
    {
        printf("epoch %d\n",epochIndex);
        float totalLoss = 0;
        if ((epochIndex + 1) % 50 == 0)
            learningRate /= 3;
		//++i是为了加速，与i++一样
		for (int i = 0; i < inputList.count; ++i)
        {
            XTensor *input = inputList.GetItem(i);
            XTensor *gold = goldList.GetItem(i);
            //改
			Forword(*input, model, net);
            //output.Dump(stderr);
			printf("Forward %d\n", i);
            XTensor loss;
            //改
			MSELoss(net.output, *gold, loss);
			printf("MSELoss %d\n", i);
			//loss.Dump(stderr);
            totalLoss += loss.Get1D(0);
			//改
            Backward(*input, *gold, model, grad, net);
			printf("Backward %d\n", i);
            //改
			Update(model, grad, learningRate);
			printf("Update %d\n", i);
            CleanGrad(grad);
			printf("ClearGrad %d\n", i);
        }
        printf("%f\n", totalLoss / inputList.count);
    }
	float tmp = model.b.Get2D(0, 0);
	model.b = NewTensor2D(model.weight1.dimSize[0], model.weight1.dimSize[1], X_FLOAT, model.devID);
	for (int i = 0; i < model.weight1.dimSize[0]; ++i) {
		for (int j = 0; j < model.weight1.dimSize[1]; ++j) {
			model.b.Set2D(tmp, i, j);
		}
	}
	
}

void Forword(XTensor &input, FNNRegModel &model, FNNRegNet &net)
{
	printf("input %d %d\n", input.dimSize[0], input.dimSize[1]);
	printf("weight1 %d %d\n", model.weight1.dimSize[0], model.weight1.dimSize[1]);
	printf("weight2 %d %d \n", model.weight2.dimSize[0], model.weight2.dimSize[1]);
	printf("b %d %d\n", model.b.dimSize[0], model.b.dimSize[1]);
    net.hidden_state1 = MatrixMul(input, model.weight1);
	printf("hidden_state1 %d %d\n", net.hidden_state1.dimSize[0], net.hidden_state1.dimSize[1]);
    net.hidden_state2 = net.hidden_state1 + model.b;
	printf("hidden_state2 %d %d\n", net.hidden_state2.dimSize[0], net.hidden_state2.dimSize[1]);
	net.hidden_state3 = HardTanH(net.hidden_state2);
	printf("hidden_state3 %d %d\n", net.hidden_state3.dimSize[0], net.hidden_state3.dimSize[1]);
	net.output = MatrixMul(net.hidden_state3, model.weight2);
	printf("output %d %d\n", net.output.dimSize[0], net.output.dimSize[1]);
}

void MSELoss(XTensor &output, XTensor &gold, XTensor &loss)
{
    XTensor tmp = output - gold;
    loss = ReduceSum(tmp, 0, 2) / output.dimSize[0];
}

void MSELossBackword(XTensor &output, XTensor &gold, XTensor &grad)
{
    XTensor tmp = output - gold;
	//tmp = ReduceSum(tmp, 0, 1) / output.dimSize[0];
    grad = tmp * 2 / output.dimSize[0];
}

void Backward(XTensor &input, XTensor &gold, FNNRegModel &model, FNNRegModel &grad, FNNRegNet &net)
{
    XTensor lossGrad;
    XTensor &dedw2 = grad.weight2;
    XTensor &dedb = grad.b;
    XTensor &dedw1 = grad.weight1;
    MSELossBackword(net.output, gold, lossGrad);
	printf("MSELossBackword\n");
    MatrixMul(net.hidden_state3, X_TRANS, lossGrad, X_NOTRANS, dedw2);
	printf("MatrixMul\n");
    XTensor dedy = MatrixMul(lossGrad, X_NOTRANS, model.weight2, X_TRANS);
	printf("dedy\n");
    _HardTanHBackward(&net.hidden_state3, &net.hidden_state2, &dedy, &dedb);
    dedw1 = MatrixMul(input, X_TRANS, dedb, X_NOTRANS);
	printf("dedw1\n");
}

void Update(FNNRegModel &model, FNNRegModel &grad, float learningRate)
{
    model.weight1 = Sum(model.weight1, grad.weight1, -learningRate);
    model.weight2 = Sum(model.weight2, grad.weight2, -learningRate);
    model.b = Sum(model.b, grad.b, -learningRate);
}

void CleanGrad(FNNRegModel &grad)
{
    grad.b.SetZeroAll();
    grad.weight1.SetZeroAll();
    grad.weight2.SetZeroAll();
}

void Test(float *testData, int testDataSize, FNNRegModel &model)
{
    FNNRegNet net;
	/*float tmp = model.b.Get2D(0, 0);
	XTensor* tmpTensor = NewTensor2D(1, );*/
	
    XTensor*  inputData = NewTensor2D(1, 1, X_FLOAT, model.devID);
    for (int i = 0; i < testDataSize; ++i)
    {

        inputData->Set2D(testData[i] / 100, 0, 0);

        Forword(*inputData, model, net);
        float ans = net.output.Get2D(0, 0) * 60;
        printf("%f\n", ans);
    }

}

};