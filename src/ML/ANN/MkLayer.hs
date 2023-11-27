module ML.ANN.MkLayer where

import Prelude as P
import Data.Array.Accelerate as A
import ML.ANN.Mat
import ML.ANN.Vect
import Data.Random.Normal
import ML.ANN.ActFuncs

type LSpec = [ActFunc]

type Weights = Mat OutputSize InputSize
type WeightsMom = Weights
type WeightsVel = Weights

type WeightsV = Vect OutputSize
type WeightsVMom = WeightsV
type WeightsVVel = WeightsV

type WeightsLR = Weights
type WeightsVLR = WeightsV

type Biases = Vect OutputSize
type BiasesMom = Biases
type BiasesVel = Biases

type BiasesLR = Biases

data Layer = SGDLayer Int Weights Biases LSpec | -- NumInputs weights bias lspec
    SGDInpLayer WeightsV Biases LSpec | -- weights bias lspec
    MomLayer Int Weights WeightsMom Biases BiasesMom LSpec | -- numInputs weights weightsMomentum biases biasesMomentum lspec
    MomInpLayer WeightsV WeightsVMom Biases BiasesMom LSpec | -- numInputs weights weightsMomentum biases biasesMomentum lspec
    RMSLayer Int Weights WeightsVel Biases BiasesVel LSpec | -- numInputs weights weightsV biases biasesV lspec
    RMSInpLayer WeightsV WeightsVVel Biases BiasesVel LSpec | -- weights weightsV biases biasesV lspec
    AdagradLayer Int Weights WeightsLR Biases BiasesLR LSpec |
    AdagradInpLayer WeightsV WeightsVLR Biases BiasesLR LSpec | -- weights weightsSummed biases biasesSummed lspec
    AdamLayer Int Weights WeightsMom WeightsVel Biases BiasesMom BiasesVel LSpec |
    AdamInpLayer WeightsV WeightsVMom WeightsVVel Biases BiasesMom BiasesVel LSpec
    deriving(Show) 

data LLayer = LLayer Layer (Vect InputSize) | -- layer previousInput
    LInpLayer Layer (Vect OutputSize)
    deriving(Show)

heWeightInit :: [Double] -> Int -> [Double]
heWeightInit randoms numInputs = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms

mkSGDInpLayer :: [Double] -> LSpec -> Layer
mkSGDInpLayer randoms lspec = do
    let numInputs = lspecGetNumOutputs lspec
        randoms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms
        weights = VectO (use (fromList (Z:.numInputs:.1) (P.take numInputs randoms2)))
        biases = weights
    SGDInpLayer weights biases lspec

mkSGDLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkSGDLayer randoms lspec numInputs numOutputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
    SGDLayer numInputs (MatOI weightsM) (VectO biasesM) lspec

mkMomInpLayer :: [Double] -> LSpec -> Int -> Layer
mkMomInpLayer randoms lspec numInputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numInputs:.1) randoms2)
        biasesM = use (fromList (Z:.numInputs:.1) randoms2)
        weightsMom = use (fromList (Z:.numInputs:.1) (P.repeat 0.0))
        biasesMom = use (fromList (Z:.numInputs:.1) (P.repeat 0.0))
    MomInpLayer (VectO weightsM) (VectO weightsMom) (VectO biasesM) (VectO biasesMom) lspec
        
mkMomLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkMomLayer randoms lspec numInputs numOutputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
        weightsMom = use (fromList (Z:.numOutputs:.numInputs) (P.repeat 0.0))
        biasesMom = use (fromList (Z:.numOutputs:.1) (P.repeat 0.0))
    MomLayer numInputs (MatOI weightsM) (MatOI weightsMom) (VectO biasesM) (VectO biasesMom) lspec

mkRMSLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkRMSLayer randoms lspec numInputs numOutputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
        weightsMom = use (fromList (Z:.numOutputs:.numInputs) (P.repeat 0.0))
        biasesMom = use (fromList (Z:.numOutputs:.1) (P.repeat 0.0))
    RMSLayer numInputs (MatOI weightsM) (MatOI weightsMom) (VectO biasesM) (VectO biasesMom) lspec

mkRMSInpLayer :: [Double] -> LSpec -> Int -> Layer 
mkRMSInpLayer randoms lspec numInputs = do
    let (MomInpLayer weights weightsMom biases biasesMom _) = mkMomInpLayer randoms lspec numInputs
    (RMSInpLayer weights weightsMom biases biasesMom lspec)

mkAdagradLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkAdagradLayer randoms lspec numInputs numOutputs = do
    let (RMSLayer _ weights weightsSummed biases biasesSummed _) = mkRMSLayer randoms lspec numInputs numOutputs
    (AdagradLayer numInputs weights weightsSummed biases biasesSummed lspec)

mkAdagradInpLayer :: [Double] -> LSpec -> Int -> Layer
mkAdagradInpLayer randoms lspec numInputs = do
    let (RMSInpLayer weights weightsSummed biases biasesSummed _) = mkRMSInpLayer randoms lspec numInputs
    (AdagradInpLayer weights weightsSummed biases biasesSummed lspec)

mkAdamLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkAdamLayer randoms lspec numInputs numOutputs = do
    let (RMSLayer _ weights weightsMom biases biasesMom _) = mkRMSLayer randoms lspec numInputs numOutputs
    AdamLayer numInputs weights weightsMom weightsMom biases biasesMom biasesMom lspec

mkAdamInpLayer :: [Double] -> LSpec -> Int -> Layer
mkAdamInpLayer randoms lspec numInputs = do
    let (RMSInpLayer weights weightsMom biases biasesMom _) = mkRMSInpLayer randoms lspec numInputs 
    AdamInpLayer weights weightsMom weightsMom biases biasesMom biasesMom lspec

lspecGetNumOutputs :: LSpec -> Int
lspecGetNumOutputs [] = 0
lspecGetNumOutputs (h : t) = (getInt h) + (lspecGetNumOutputs t)

