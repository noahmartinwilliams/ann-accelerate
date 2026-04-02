module ML.ANN.LSpec where

import ML.ANN.Types
import Prelude as P

getLSpecNumOuts :: LSpec -> Int
getLSpecNumOuts l = P.foldr (+) 0 (P.map P.fst l)
