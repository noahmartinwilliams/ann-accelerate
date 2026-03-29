{-# LANGUAGE FlexibleContexts #-}
module ML.ANN.ErrorFn where

import Data.Array.Accelerate as A
import Prelude as P

mseErrorFn :: (Shape sh, Elt e, A.Num e) => Acc (Array sh e) -> Acc (Array sh e) -> Acc (Array sh e)
mseErrorFn x y = let diff = A.zipWith (-) x y in A.zipWith (*) diff diff 

dmseErrorFn :: (Shape sh, Elt e, A.Num e) => Acc (Array sh e) -> Acc (Array sh e) -> Acc (Array sh e)
dmseErrorFn y x = let diff = A.zipWith (-) y x in A.zipWith (+) diff diff
