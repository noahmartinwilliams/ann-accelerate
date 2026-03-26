module ML.ANN.LLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.InfLayer
import ML.ANN.Types
import Prelude as P

learnLayer :: Layer -> Acc (Matrix Double) -> (LLayer, Acc (Matrix Double))
learnLayer layer inp = do
    let inferred = inferLayer layer inp
    (LLayer { llprevInput = (AccMat inp Inp One), llayer = layer }, inferred)

batchLearnLayers :: Layer -> [Acc (Matrix Double)] -> ([LLayer], [Acc (Matrix Double)])
batchLearnLayers l inpLs = P.unzip (P.map (learnLayer l) inpLs)
