module ML.ANN.Network where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.ANN.Block
import ML.ANN.BPLayer
import ML.ANN.ErrorFn
import ML.ANN.InfLayer
import ML.ANN.LLayer
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P
import System.Random

mkNetwork :: StdGen -> [LSpec] -> Optim -> ErrorFn -> Network
mkNetwork gen (first : lspecLs) optim errf = do
    let norms = normals gen
        (inpLayer, norms') = mkInpLayer first norms 
    (Network (inpLayer : (restOfLayers first lspecLs norms')) optim errf) where

        numIns :: LSpec -> Int
        numIns l = P.foldr (+) 0 (P.map P.fst l)

        restOfLayers :: LSpec -> [LSpec] -> [Double] -> [Layer]
        restOfLayers _ [] _ = []
        restOfLayers lspec [lspec'] rands = let (l, _) = mkLayer (numIns lspec) lspec' rands in [l]
        restOfLayers lspec (lspec' : rest) rands = do
            let (l, rands') = mkLayer (numIns lspec) lspec' rands
            l : (restOfLayers lspec' rest rands')


networkGetErrorFn :: Network -> ErrorFn
networkGetErrorFn (Network _ _ e) = e

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

batchLearnNetwork :: Network -> [Acc (Matrix Double)] -> ([LNetwork], [Acc (Matrix Double)])
batchLearnNetwork n inpLs = P.unzip (P.map (learnNetwork n) inpLs)


bpNetwork :: LNetwork -> Acc (Matrix Double) -> (Network, Acc (Matrix Double))
bpNetwork (LNetwork layers optim errfn) bp = do
    let bp' = AccMat bp Outp One
        (n, (AccMat e Outp One)) = intern (P.reverse layers) bp' optim 
    (n, e) where

        intern :: [LLayer] -> AccMat Double Outp One -> Optim -> (Network, AccMat Double Outp One)
        intern [] a o = ((Network [] o errfn), a)
        intern ( h : t) a opt = do
            let (l, e) = bpLayer h opt a
                ((Network l' _ errfn), e') = intern t e opt
            (Network (l' P.++ [l]) opt errfn, e')

trainOnce :: BLInfo -> AccBlock -> Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double, Vector Int, Vector Double)
trainOnce blinfo block sample = do
    let net = block2network blinfo block
        (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        (ln, netOut) = learnNetwork net inp
        (errFn, derrFn) = networkGetErrorFn net
        err = errFn netOut outp
        derr = derrFn netOut outp
        (net', bp) = bpNetwork ln derr
        (_, block') = network2block net'
        (blockI, blockD) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        ret = A.lift (err, bp, blockI, blockD)
    ret

trainMiniBatch :: Int -> BLInfo -> AccBlock -> Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double, Vector Int, Vector Double)
trainMiniBatch miniSize blinfo block sample = do
    let net = block2network blinfo block
        (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        (nets, outs, bps) = trainIntern miniSize net inp outp
        net' = avgNets nets
        avg x = A.map (\y -> y / (constant (P.fromIntegral miniSize :: Double))) (A.sum x)
        outsAvged = avg outs
        bpsAvged = A.replicate (constant (Z:.All:.(1 :: Int))) (avg bps)
        (err, _) = networkGetErrorFn net'
        err' = (A.replicate (constant (Z:.All:.(1 :: Int))) (A.sum (err outs outp))) 
        (_, block') = network2block net'
        (blockI, blockD) = A.unlift block' :: (Acc (Vector Int), Acc (Vector Double))
    A.lift (err', bpsAvged, blockI, blockD) where
        
        trainIntern :: Int -> Network -> Acc (Matrix Double) -> Acc (Matrix Double) -> ([Network], Acc (Matrix Double), Acc (Matrix Double))
        trainIntern 1 net inp outp = do
            let inp' = A.take (constant 1) inp
                outp' = A.take (constant 1) outp
                (blinfo, block) = network2block net
                trained = trainOnce blinfo block (A.lift (inp', outp'))
                (err, bp, vi, vd) = A.unlift trained :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int), Acc (Vector Double))
            ([(block2network blinfo (A.lift (vi, vd)))], err, bp)
        trainIntern ms net inp outp = do
            let inp' = A.take (constant 1) inp
                outp' = A.take (constant 1) outp
                (blinfo, block) = network2block net
                trained = trainOnce blinfo block (A.lift (inp', outp'))
                (err, bp, vi, vd) = A.unlift trained :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int), Acc (Vector Double))
                net' = block2network blinfo (A.lift (vi, vd))
                (nets, err', bp') = trainIntern (ms - 1) net (A.drop (constant 1) inp) (A.drop (constant 1) outp)
            (net' : nets , err A.++ err' , bp A.++ bp')


avgNets :: [Network] -> Network
avgNets (h : ns) = let added = P.foldr addNets h ns in scaleNet (constant (1.0 / (P.fromIntegral (P.length (h : ns)) :: Double))) added

scaleNet :: Exp Double -> Network -> Network
scaleNet s (Network ls o e) = Network (P.map (scaleLayer s) ls) o e

scaleLayer :: Exp Double -> Layer -> Layer
scaleLayer s l@(InpLayer { vweights = vw, vbiases = vb, vbiasesMom = vbm, vbiasesVel = vbv, vweightsMom = vwm, vweightsVel=vwv}) = l { vweights = (matScale s vw), vbiases = (matScale s vb), vbiasesMom = (s `matScale` vbm), vbiasesVel = (s `matScale` vbv) , vweightsMom = (s `matScale` vwm), vweightsVel = (s `matScale` vwv)}
scaleLayer s l@(Layer { lweights = lw, lbiases = lb, lbiasesMom = lbm, lbiasesVel = lbv, lweightsMom = lwm, lweightsVel = lwv}) = l { lweights = (matScale s lw), lbiases = (matScale s lb), lbiasesMom = (s `matScale` lbm), lbiasesVel = (s `matScale` lbv), lweightsMom = (s `matScale` lwm), lweightsVel = (s `matScale` lwv)}

addNets :: Network -> Network -> Network
addNets (Network layers o e) (Network layers' o' e') = do
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
