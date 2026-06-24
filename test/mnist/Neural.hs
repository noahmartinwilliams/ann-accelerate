module Neural where

import Debug.Trace
import Conf
import Control.Monad.Reader
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.LLVM.PTX as PTX
import qualified Data.ByteString as B
import Data.List.Split
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import Samps
import System.Random
import Text.Printf

runNeural :: Int -> Int -> B.ByteString -> B.ByteString -> B.ByteString -> B.ByteString -> Reader Conf (String, String, String, String)
runNeural seed lineNo imgs answers testImgs testAnswers = do
    let gen = mkStdGen seed
    cnf <- ask
    epochs <- reader numEpochs
    neural <- getNeural gen ( optimizer cnf)
    mbs <- reader miniBatchSize
    let imgs' = B.drop 16 imgs
        answers' = B.drop 8 answers
        testImgs' = B.drop 16 imgs
        testAnswers' = B.drop 8 testAnswers
        retName = "/tmp/results/errs-" P.++ (show lineNo) P.++ ".txt"
        retName' = "/tmp/results/test-" P.++ (show lineNo) P.++ ".txt"
        samps = mkSamps mbs imgs' answers'
        samps' = mkSamps 1 testImgs' testAnswers'
        (blinfo, block) = network2block neural
        block' = I.run block
        fn = PTX.runN (trainMiniBatch mbs blinfo )
        (errs, vi, vd) = runNeural' (P.length samps) epochs fn block' samps
    if errs P.== []
    then
        return (retName, "", retName', "")
    else do
        let (err' : errs') = P.map (showErrs) errs
            retStr = P.foldl (\x -> \y -> x P.++ "\n" P.++ y) err' errs'
            net' = block2network blinfo (use (vi, vd))
            testRes = testResults net' samps'
        if testRes P.== []
        then
            return (retName, retStr, retName', "")
        else do
            let (err'' : errs'') = P.map (printf "%.5f") testRes
                retStr' = P.foldl (\x -> \y -> x P.++ "\n" P.++ y)  err'' errs''
            return (retName, retStr, retName', retStr')

showErrs :: [Double] -> String
showErrs l = do
    let (l'' : l') = l
    let (lStr : lRest) = P.map (printf "%.5f") l'
        lCommas = P.foldl (\x -> \y -> x P.++ "," P.++ y) lStr lRest
    (printf "%.5f" l'') P.++ "," P.++ lCommas

testResults :: Network -> [(Matrix Double, Matrix Double)] -> [Double]
testResults net samps = do
    let fn = PTX.runN (inferNetwork net) 
        (inps, outps) = P.unzip samps
        outps' = P.map (A.toList) outps
        results = P.map fn inps
        results' = P.map (A.toList) results
        err a b = P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) a b)
        errs = P.map (\(x, y) -> (err x y) / 10.0) (P.zip outps' results')
    P.takeWhile (\x -> (P.isNaN x) P.== False) errs

getNeural :: StdGen -> String -> Reader Conf Network
getNeural g "Adam" = do
    cnf <- ask
    lr1 <- reader lr
    b1 <- reader beta1
    b2 <- reader beta2
    mbs <- reader miniBatchSize
    let errFn = getErrorFn (costF cnf)
        lsp = read (layers cnf) :: [LSpec]
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (AdamOptim (constant lr1) (constant b1) (constant b2)) errFn
    return net

getNeural g "SGD" = do
    cnf <- ask
    lr1 <- reader lr
    errFn <- reader costF
    mbs <- reader miniBatchSize
    let lsp = read (layers cnf) :: [LSpec]
        errFn' = getErrorFn errFn
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (SGDOptim (constant lr1)) errFn'
    return net

runNeural' :: Int -> Int -> Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([[Double]], Vector Int, Vector Double)
runNeural' numSamps 1 fn block samples = runner fn block samples
runNeural' numSamps i fn block samples = do
    let (errs, bi, bd) = (runner fn block samples)
        (errs', bi', bd') = runNeural' numSamps (i - 1) fn (bi, bd) samples
    if P.isNaN ((errs P.!! 0) P.!! 0)
    then
        (errs, bi, bd)
    else
        (errs P.++ errs', bi', bd') 

runner :: Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([[Double]], Vector Int, Vector Double)
runner fn bl [last] = do
    let (errs, vi, vd) = fn bl last 
        l = A.toList errs
        summed = P.sum l
    if P.isNaN summed
    then
        ([], vi, vd)
    else
        ([summed : l], vi, vd) 
        
runner fn bl (first : rest) = do
    let (err, vi, vd) = fn bl first
        (err', vi', vd') = runner fn (vi, vd) rest
        errL = (A.toList err)
    if P.isNaN (P.sum errL)
    then
        ([], vi, vd)
    else
        (((P.sum errL) : errL) : err', vi', vd')

getErrorFn :: String -> ErrorFn
getErrorFn "MSE" = (mseErrorFn, dmseErrorFn)
getErrorFn "CrossEntropy" = (crossEntropyErrorFn, dcrossEntropyErrorFn)
