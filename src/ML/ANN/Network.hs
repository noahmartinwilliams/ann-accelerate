module ML.ANN.Network where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.ANN.Block
import ML.ANN.BPLayer
import ML.ANN.ErrorFn
import ML.ANN.InfLayer
import ML.ANN.LLayer
import ML.ANN.LSpec
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P
import System.Random

mkNetwork :: StdGen -> [LSpec] -> Optim -> ErrorFnT -> Network
mkNetwork gen (first : lspecLs) optim errf = do
    let norms = normals gen
        (inpLayer, norms') = mkInpLayer first norms 
    (Network (inpLayer : (restOfLayers lspecLs norms' (numOuts first ))) optim errf) where

        numOuts :: LSpec -> Int
        numOuts l = P.foldr (+) 0 (P.map P.fst l)

        restOfLayers :: [LSpec] -> [Double] -> Int -> [Layer]
        restOfLayers  [] _ _ = []
        restOfLayers [lspec'] rands numIns = let (l, _) = mkLayer numIns lspec' rands in [l]
        restOfLayers (lspec' : rest) rands numIns = do
            let (l, rands') = mkLayer numIns lspec' rands
            l : (restOfLayers rest rands' (numOuts lspec'))


networkGetErrorFn :: Network -> ErrorFn
networkGetErrorFn (Network _ _ e) = lookupErrorFnT e

inferNetwork :: Network -> Acc (Matrix Double) -> Acc (Matrix Double)
inferNetwork (Network [l] _ _ ) x = inferLayer l x
inferNetwork (Network (l : rest) o errfn) x = do
    let y = inferLayer l x
    inferNetwork (Network rest o errfn) y

learnNetwork :: Network -> Acc (Matrix Double) -> (LNetwork, Acc (Matrix Double))
learnNetwork (Network [] optim errfn) m = (LNetwork [] optim errfn, m)
learnNetwork (Network (h : t) optim errfn) m = do
    let (l, m') = learnLayer h m
        ((LNetwork l' _ errfn'), m'') = learnNetwork (Network t optim errfn) m'
    (LNetwork (l : l') optim errfn', m'')

bpNetwork :: LNetwork -> Acc (Matrix Double) -> (Network, Acc (Matrix Double))
bpNetwork (LNetwork layers optim errfn) bp = do
    let bp' = AccMat bp Outp One
        (n, (AccMat e Outp One)) = intern (P.reverse layers) bp' optim 
    (n, e) where

        intern :: [LLayer] -> AccMat Double Outp One -> Optim -> (Network, AccMat Double Outp One)
        intern [] a o = ((Network [] o errfn), a)
        intern ( h : t) a opt = do
            let (l, e) = bpLayer h opt a
                ((Network l' _ errfn'), e') = intern t e opt
            (Network (l' P.++ [l]) opt errfn', e')

trainOnce :: BLInfo -> AccBlock -> Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double, (Vector Int, Vector Double))
trainOnce blinfo block sample = do
    let net = block2network blinfo block
        (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        (ln, netOut) = learnNetwork net inp
        (errFn, derrFn) = networkGetErrorFn net
        err = errFn netOut outp
        derr = derrFn netOut outp
        (net', bp) = bpNetwork ln derr
        (_, block') = network2block net'
        block'' = A.unlift block' :: (Acc (Vector Int), Acc (Vector Double))
        ret = A.lift (err, bp, block'')
    ret

type AccInp = Acc (Matrix Double)
type AccOutp = Acc (Matrix Double)
type InpA = Matrix Double
type OutpA = Matrix Double
type AccErrs = Acc (Matrix Double)
type ErrsA = Matrix Double
type HypParamsA = Vector Double
type AccHypParams = Acc (Vector Double)
type IntHypParamsA = Vector Int
type AccIntHypParams = Acc (Vector Int)
type ParamsA = Matrix Double
type AccParams = Acc (Matrix Double)

type AccNetState = Acc (ErrsA, InpA, OutpA, IntHypParamsA, HypParamsA, ParamsA)
type NetStateAcc = (AccErrs, AccInp, AccOutp, AccIntHypParams, AccHypParams, AccParams)

trainMiniBatch :: Int -> BLInfo -> AccBlock -> Acc (Matrix Double, Matrix Double) -> Acc (Vector Double, (Vector Int, Vector Double))
trainMiniBatch 1 blinfo block sample = do
    let (a', _, c) = A.unlift (trainOnce blinfo block sample) :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double))
    A.lift (A.flatten a', c)
trainMiniBatch miniSize blinfo block sample = do
    let (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        numEnds = getBlockNumOuts blinfo
        empty = A.fromList (Z:.numEnds:.1) (P.take numEnds (P.repeat 0.0)) :: (Matrix Double)
        block' = incNumTimes block
        (hi, hd, p) = splitHypParams blinfo block'
        netState = A.lift (empty, inp, outp, hi, hd, A.replicate (A.lift (Z:.All:.(1::Int))) p)
        ret = awhile test (trainOnce' blinfo) netState  
        (errs, _, _, hi', hd', p') = A.unlift ret :: NetStateAcc
        divideMS = A.map (\x -> x / (constant (P.fromIntegral miniSize :: Double)))
        p'' = divideMS (A.sum p')
        block'' = A.lift (hi', hd' A.++ p'') :: AccBlock
    A.lift ((A.sum errs), block'') where

        test :: AccNetState -> Acc (Scalar Bool)
        test bl = do
            let (_, inps, _, _, _, _) = A.unlift bl :: NetStateAcc
            A.unit (A.not (A.null inps))
        
        trainOnce' :: BLInfo -> AccNetState -> AccNetState
        trainOnce' blinfo' netState' = do
            let (errs, inp, outp, intHypParams, hypParams, params) = A.unlift netState' :: NetStateAcc
                newBlock = A.lift (intHypParams, (hypParams A.++ (A.flatten (A.take (constant 1) params))))
                trained = trainOnce blinfo' newBlock (A.lift (inp, outp))
                (err, _, newBlock') = A.unlift trained :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double))
                (_, _, params') = splitHypParams blinfo' newBlock'
                inp'' = A.drop (constant 1) inp
                outp'' = A.drop (constant 1) outp
                errs' = err A.++ errs
            A.lift (errs', inp'', outp'', intHypParams, hypParams, (A.replicate (A.lift (Z:.All:.(1::Int))) params') A.++ params)

combineBlocks :: Acc (Vector Int, Vector Double) -> Acc (Vector Int, Vector Double) -> Acc (Vector Int, Vector Double)
combineBlocks a b = do
    let (_, ad) = A.unlift a :: (Acc (Vector Int), Acc (Vector Double))
        (_, bd) = A.unlift b :: (Acc (Vector Int), Acc (Vector Double))
        apb = A.zipWith (+) ad bd
        summed = A.sum apb
    acond (A.the (A.map A.isNaN summed)) a b 

scaleNet :: Exp Double -> Network -> Network
scaleNet s (Network ls o e) = Network (P.map (scaleLayer s) ls) o e

scaleLayer :: Exp Double -> Layer -> Layer
scaleLayer s l@(InpLayer { vweights = vw, vbiases = vb, vbiasesMom = vbm, vbiasesVel = vbv, vweightsMom = vwm, vweightsVel=vwv}) = l { vweights = (matScale s vw), vbiases = (matScale s vb), vbiasesMom = (s `matScale` vbm), vbiasesVel = (s `matScale` vbv) , vweightsMom = (s `matScale` vwm), vweightsVel = (s `matScale` vwv)}
scaleLayer s l@(Layer { lweights = lw, lbiases = lb, lbiasesMom = lbm, lbiasesVel = lbv, lweightsMom = lwm, lweightsVel = lwv}) = l { lweights = (matScale s lw), lbiases = (matScale s lb), lbiasesMom = (s `matScale` lbm), lbiasesVel = (s `matScale` lbv), lweightsMom = (s `matScale` lwm), lweightsVel = (s `matScale` lwv)}

addNets :: Network -> Network -> Network
addNets (Network layers o e) (Network layers' _ _) = do
    let layers'' = P.zipWith addLayer layers layers'
    (Network layers'' o e)

addLayer :: Layer -> Layer -> Layer
addLayer i@(InpLayer { vweights = vw, vbiases = vb, vweightsMom = vwm, vbiasesMom = vbm, vweightsVel = vwv, vbiasesVel = vbv}) (InpLayer { vweights = vw', vbiases = vb', vweightsMom = vwm', vbiasesMom = vbm', vweightsVel = vwv', vbiasesVel = vbv'}) = do
    let vw'' = vw `matAdd` vw'
        vb'' = vb `matAdd` vb'
        vwm'' = vwm `matAdd` vwm'
        vbm'' = vbm `matAdd` vbm'
        vbv'' = vbv `matAdd` vbv'
        vwv'' = vwv `matAdd` vwv'
    i { vweights = vw'', vbiases = vb'', vweightsMom = vwm'', vweightsVel = vwv'', vbiasesMom = vbm'', vbiasesVel = vbv'' }

addLayer i@(Layer { lweights = lw, lbiases = lb, lweightsMom = lwm, lweightsVel = lwv, lbiasesMom = lbm, lbiasesVel = lbv}) (Layer { lweights = lw', lbiases = lb', lweightsVel = lwv', lweightsMom = lwm', lbiasesVel = lbv', lbiasesMom = lbm' }) = do
    let lw'' = lw `matAdd` lw'
        lb'' = lb `matAdd` lb'
        lwm'' = lwm `matAdd` lwm'
        lbm'' = lbm `matAdd` lbm'
        lwv'' = lwv `matAdd` lwv'
        lbv'' = lbv `matAdd` lbv'
    i { lweights = lw'', lbiases = lb'', lweightsMom = lwm'', lweightsVel = lwv'', lbiasesMom = lbm'', lbiasesVel = lbv''}

getBlockNumOuts :: BLInfo -> Int
getBlockNumOuts (BLAdam blinfo _) = do
    let ((LayerInfo _ last _) : _) = P.reverse blinfo
        numOuts = getLSpecNumOuts last
    numOuts
getBlockNumOuts (BLSGD blinfo _) = do
    let ((LayerInfo _ last _) : _) = P.reverse blinfo
        numOuts = getLSpecNumOuts last
    numOuts
