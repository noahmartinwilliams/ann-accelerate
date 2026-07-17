module Neural where

import Control.Monad.Reader
import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import Data.ByteString as B
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.Random
import Samps
import Types

getNeural :: StdGen -> Mon ()
getNeural g = do
    errfn <- reader costF
    opt <- reader optimizer
    mbs <- reader miniBatchSize
    lyrs <- reader layers
    iaf <- reader inputAF
    learnRate <- reader lr
    b1 <- reader beta1
    b2 <- reader beta2
    let iaf' = getAF iaf
    let n = mkNetwork g ([[(28*28, iaf')] ] P.++ (getLayers lyrs) P.++ [[(10, SoftMax)]]) (getOptim opt learnRate b1 b2) (getErrfn errfn) 
        (blinfo, ablock) = network2block n
        block = run ablock
    modify (\s -> s { stBlock = block, stBLInfo = blinfo})

getLayers :: String -> [LSpec]
getLayers str = read str :: [LSpec]

getErrfn :: String -> ErrorFnT
getErrfn "MSE" = MSEErrorFn
getErrfn "CrossEntropy" = CrossEntropyErrorFn

getOptim :: String -> Double -> Double -> Double -> Optim
getOptim "SGD" lr _ _ = SGDOptim (constant lr)
getOptim "Adam" lr b1 b2 = AdamOptim (constant lr) (constant b1) (constant b2)

getAF :: String -> ActFunc
getAF str = read str :: ActFunc
