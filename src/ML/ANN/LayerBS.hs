module ML.ANN.LayerBS where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import Data.Array.Accelerate.Matrix
import Data.Binary
import Data.ByteString as B
import Data.Word
import ML.ANN.BString
import ML.ANN.Types
import Prelude as P

instance Binary Network where
    put (Network layers optim errorfn) = do
        put (0x7F :: Word8)
        put (65 :: Word8) -- A
        put (78 :: Word8) -- N
        put (78 :: Word8) -- N
        put (0 :: Word8)
        layers2bs layers optim 
        optim2bs optim
        errorFnT2bs errorfn

    get = do
        _ <- getWord8
        _ <- getWord8
        _ <- getWord8
        _ <- getWord8
        _ <- getWord8
        nl <- get 
        ls <- bs2layers nl
        optim <- bs2optim
        errfn <- bs2errorFn
        return (Network ls optim errfn)

layer2bs :: Layer -> Optim -> Put
layer2bs (InpLayer {vweights = vw, vbiases = vb, vlspec = vls}) (SGDOptim _) = do
    put (0 :: Word8) -- optimizer
    put (0 :: Word8) -- layer type

    biases2bs vw
    biases2bs vb
    lspec2bs vls

layer2bs (Layer { lnumInputs = lni, lweights = lw, lbiases = lb, llspec = lsp}) (SGDOptim _) = do
    put (0 :: Word8)
    put (1 :: Word8)

    let (AccMat lb' _ _ ) = lb
        lno = P.length (A.toList (run lb'))
    put lno
    put lni
    weights2bs (lw, lno, lni)
    biases2bs lb 
    lspec2bs lsp

layer2bs l@(InpLayer { vnumTimes = vnt, vweights = vw, vbiases = vb, vlspec = vls}) (AdamOptim _ _ _) = do
    let vwm = vweightsMom l
        vbm = vbiasesMom l
        vwv = vweightsVel l
        vbv = vbiasesVel l
    put (1 :: Word8)
    put (0 :: Word8)

    eint2bs (vnumTimes l)
    biases2bs vw
    biases2bs vb
    biases2bs vwm
    biases2bs vbm
    biases2bs vwv
    biases2bs vbv
    lspec2bs vls

layer2bs l@(Layer { lnumTimes = lnt, lweights = lw, lbiases = lb, llspec = lls}) (AdamOptim _ _ _) = do
    let lwm = lweightsMom l
        lbm = lbiasesMom l
        lwv = lweightsVel l
        lbv = lbiasesVel l
        (AccMat lb' _ _ ) = lb
        lno = P.length (A.toList (run lb'))
        lni = lnumInputs l
    put (1 :: Word8)
    put (1 :: Word8)

    eint2bs lnt
    weights2bs (lw, lni, lno)
    biases2bs lb
    weights2bs (lwm, lni, lno)
    biases2bs lbm
    weights2bs (lwv, lni, lno)
    biases2bs lbv
    lspec2bs lls

layers2bs :: [Layer] -> Optim -> Put
layers2bs ls o = P.mapM_ (\x -> layer2bs x o) ls

bs2layers :: Int -> Get [Layer]
bs2layers 0 = return []
bs2layers i = do
    l <- bs2layer
    r <- bs2layers (i - 1)
    return (l : r)

bs2layer :: Get Layer
bs2layer = do
    otype <- getWord8 
    ltype <- getWord8 
    case (otype, ltype) of
        (0, 0) -> do
            vw <- bs2biases
            vb <- bs2biases
            vls <- bs2lspec
            let zeros = use (A.fromList (Z:.1:.1) [0.0])
                zerosA = AccMat zeros Outp One
            return (InpLayer { vnumTimes = constant 1, vweights = vw, vbiases = vb, vlspec = vls, vweightsMom = zerosA, vbiasesMom = zerosA, vweightsVel = zerosA, vbiasesVel = zerosA})

        (0, 1) -> do
            (lw, _, _ ) <- bs2weights
            lb <- bs2biases
            lls <- bs2lspec
            let zeros = use (A.fromList (Z:.1:.1) [0.0])
                zerosAM = AccMat zeros Outp Inp
                zerosAV = AccMat zeros Outp One
                (AccMat lb' _ _) = lb
                (AccMat lw' _ _ ) = lw
                lno = P.length (A.toList (run lb'))
                lni = div (P.length (A.toList (run lw'))) lno
            return (Layer { lnumInputs = lni, lnumTimes = constant 1, lweights = lw, lbiases = lb, llspec = lls, lweightsMom = zerosAM, lbiasesMom = zerosAV, lweightsVel = zerosAM, lbiasesVel = zerosAV})
        (1, 0) -> do
            nt <- bs2eint
            vw <- bs2biases
            vb <- bs2biases
            vwm <- bs2biases
            vbm <- bs2biases
            vwv <- bs2biases
            vbv <- bs2biases
            lsp <- bs2lspec 
            return (InpLayer { vnumTimes = nt, vweights = vw, vbiases = vb, vlspec = lsp, vweightsMom = vwm, vbiasesMom = vbm, vweightsVel = vwv, vbiasesVel = vbv})

        (1, 1) -> do
            nt <- bs2eint
            (lw, _, _) <- bs2weights
            lb <- bs2biases
            (lwm, _, _) <- bs2weights
            lbm <- bs2biases
            (lwv, _, _) <- bs2weights
            lbv <- bs2biases
            lsp <- bs2lspec 
            let (AccMat lb' _ _) = lb
            let ni = P.length (A.toList (run lb'))
            return (Layer { lnumInputs = ni, lnumTimes = nt, lweights = lw, lbiases = lb, llspec = lsp, lweightsMom = lwm, lbiasesMom = lbm, lweightsVel = lwv, lbiasesVel = lbv})
