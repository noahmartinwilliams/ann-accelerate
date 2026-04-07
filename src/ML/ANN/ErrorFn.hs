{-# LANGUAGE FlexibleContexts #-}
module ML.ANN.ErrorFn where

import Data.Array.Accelerate as A
import Prelude as P

mseErrorFn :: (Shape sh) => Acc (Array sh Double) -> Acc (Array sh Double) -> Acc (Array sh Double)
mseErrorFn x y = do
    let diff = A.zipWith (-) x y 
        len = A.length (A.flatten x)
    A.map (\x -> x / (A.fromIntegral len :: Exp Double)) (A.zipWith (*) diff diff )

dmseErrorFn :: (Shape sh) => Acc (Array sh Double) -> Acc (Array sh Double) -> Acc (Array sh Double)
dmseErrorFn y x = do
    let diff = A.zipWith (-) y x 
        len = A.length (A.flatten x)
    A.map (\x -> x / (A.fromIntegral len :: Exp Double)) (A.zipWith (+) diff diff )
