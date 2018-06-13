// Learn more about F# at http://fsharp.org
open System
open Microsoft.ML
open Microsoft.ML.Runtime
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Data
open Microsoft.ML.Transforms
open Microsoft.ML.Trainers


type IrisData() =
    
    [<Column(ordinal = "0");DefaultValue>] val mutable public SepalLength: float32
    [<Column(ordinal = "1");DefaultValue>] val mutable public SepalWidth: float32
    [<Column(ordinal = "2");DefaultValue>] val mutable public PetalLength:float32
    [<Column(ordinal = "3");DefaultValue>] val mutable public PetalWidth:float32    
    [<Column(ordinal = "4",name="Label");DefaultValue>] val mutable public Label: string


type IrisPrediction() =
    [<ColumnName "PredictedLabel";DefaultValue>] val mutable public PredictedLabel : string

[<EntryPoint>]
let main argv =

    let dataPath = "./iris-data.txt"
    
    // Initialize Compute Graph
    let pipeline = new LearningPipeline()
    
    // Load Data
    pipeline.Add((new TextLoader(dataPath)).CreateFrom<IrisData>(separator=','))

    // Transform Data
    // Assign numeric values to text in the "Label" column, because 
    // only numbers can be processed during model training 
    pipeline.Add(new Transforms.Dictionarizer("Label"))

    // Vectorize Features
    pipeline.Add(new ColumnConcatenator("Features","SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))

    // Add Learner
    pipeline.Add(new StochasticDualCoordinateAscentClassifier())

    // Convert Label back to text 
    pipeline.Add(new Transforms.PredictedLabelColumnOriginalValueConverter(PredictedLabelColumn = "PredictedLabel"))

    //Train the model
    let model = pipeline.Train<IrisData, IrisPrediction>()

    // Test data for prediction
    let testInstance = IrisData()
    testInstance.SepalLength <- 3.3f
    testInstance.SepalWidth <- 1.6f
    testInstance.PetalLength <- 0.2f
    testInstance.PetalWidth <- 5.1f

    //Get Prediction
    let prediction = model.Predict(testInstance)

    //Output Prediction
    printfn "Predicted flower type is: %s" prediction.PredictedLabel
    0 // return an integer exit code