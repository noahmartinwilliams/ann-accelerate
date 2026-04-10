module Neural where

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
    neural <- getNeural gen ( optimizer cnf)
    let imgs' = B.drop 16 imgs
        answers' = B.drop 8 answers
        testImgs' = B.drop 16 imgs
        testAnswers' = B.drop 8 testAnswers
        samps = mkSamps (miniBatchSize cnf) imgs' answers'
        samps' = mkSamps 1 testImgs' testAnswers'
        mbs = miniBatchSize cnf
        epochs = numEpochs cnf
        (blinfo, block) = network2block neural
        block' = I.run block
        fn = PTX.runN (trainMiniBatch mbs blinfo )
        (errs, _, vi, vd) = runNeural' epochs fn block' samps
        (err' : errs') = P.map (showErrs) errs
        retStr = P.foldl (\x -> \y -> x P.++ "\n" P.++ y) err' errs'
        retName = "/tmp/results/errs-" P.++ (show lineNo) P.++ ".txt"
        retName' = "/tmp/results/test-" P.++ (show lineNo) P.++ ".txt"
        net' = block2network blinfo (use (vi, vd))
        testRes = testResults net' samps'
        (err'' : errs'') = P.map (printf "%.5f") testRes
        retStr' = P.foldl (\x -> \y -> x P.++ "\n" P.++ y)  err'' errs''
    (return (retName, retStr, retName', retStr'))

showErrs :: [Double] -> String
showErrs l = do
    let (lStr : lRest) = P.map (printf "%.5f") l
        lCommas = P.foldl (\x -> \y -> x P.++ "," P.++ y) lStr lRest
    lCommas

testResults :: Network -> [(Matrix Double, Matrix Double)] -> [Double]
testResults net samps = do
    let fn = PTX.runN (inferNetwork net) 
        (inps, outps) = P.unzip samps
        outps' = P.map (A.toList) outps
        results = P.map fn inps
        results' = P.map (A.toList) results
        err a b = P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) a b)
        errs = P.map (\(x, y) -> (err x y) / 10.0) (P.zip outps' results')
    errs

getNeural :: StdGen -> String -> Reader Conf Network
getNeural g "Adam" = do
    cnf <- ask
    let lr1 = lr cnf
        b1 = beta1 cnf
        b2 = beta2 cnf
        errFn = getErrorFn (costF cnf)
        mbs = miniBatchSize cnf
        lsp = read (layers cnf) :: [LSpec]
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (AdamOptim (constant lr1) (constant b1) (constant b2)) errFn
    return net

getNeural g "SGD" = do
    cnf <- ask
    let lr1 = lr cnf
        errFn = getErrorFn (costF cnf)
        mbs = (miniBatchSize cnf)
        lsp = read (layers cnf) :: [LSpec]
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (SGDOptim (constant lr1)) errFn
    return net

runNeural' :: Int -> Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([[Double]], [Matrix Double], Vector Int, Vector Double)
runNeural' i fn block samples = do
    if i P.== 1
    then
        (runner fn block samples) 
    else do
        let (errs, bps, bi, bd) = runner fn block samples
            (errs', bps', bi', bd') = runNeural' (i - 1) fn (bi, bd) samples
        (errs P.++ errs', bps P.++ bps', bi', bd') where

            runner :: Fn -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([[Double]], [Matrix Double], Vector Int, Vector Double)
            runner fn bl [last] = do
                let (errs, bps, vi, vd) = fn bl last 
                    l = A.toList errs
                ([P.sum l : l], [bps], vi, vd) 
                    
            runner fn bl (first : rest) = do
                let (err, bp, vi, vd) = fn bl first
                    (err', bp', vi', vd') = runner fn (vi, vd) rest
                    errL = ((A.toList err))
                (((P.sum errL) : errL) : err', bp : bp', vi', vd')

getErrorFn :: String -> ErrorFn
getErrorFn "MSE" = (mseErrorFn, dmseErrorFn)
getErrorFn "CrossEntropy" = (crossEntropyErrorFn, dcrossEntropyErrorFn)
