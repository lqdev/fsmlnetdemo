# Classification with F# ML.NET Models

This is a sample F# ML.NET classification model ported over from another C# [project](https://github.com/lqdev/mlnetacidemo). A more detailed blog post with instructions can be found at the following [link](http://luisquintanilla.me/2018/06/13/mlnet-classification-fsharp/)

## Build Project

```bash
git clone https://github.com/lqdev/fsmlnetdemo.git
cd fsmlnetdemo
dotnet restore
dotnet build
```

## Run Project

```bash
dotnet run -p model/model.fsproj
```

Sample Output:

```bash
Automatically adding a MinMax normalization transform, use 'norm=Warn' or 'norm=No' to turn this behavior off.
Using 2 threads to train.
Automatically choosing a check frequency of 2.
Auto-tuning parameters: maxIterations = 9998.
Auto-tuning parameters: L2 = 2.667734E-05.Auto-tuning parameters: L1Threshold (L1/L2) = 0.
Using best model from iteration 1066.Not training a calibrator because it is not needed.
Predicted flower type is: Iris-virginica
```