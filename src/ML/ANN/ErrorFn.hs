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

crossEntropyErrorFn :: (Shape sh) => Acc (Array sh Double) -> Acc (Array sh Double) -> Acc (Array sh Double)
crossEntropyErrorFn realAnswer expectedAnswer = do
    let len = A.length (A.flatten realAnswer)
        one = constant 1.0
        sum = (A.zipWith (\a -> \b -> a * (log b) + (one-a)*(log (one - b))) expectedAnswer realAnswer)
    A.map (\x -> - x / (A.fromIntegral len :: Exp Double)) sum

dcrossEntropyErrorFn :: (Shape sh) => Acc (Array sh Double) -> Acc (Array sh Double) -> Acc (Array sh Double)
dcrossEntropyErrorFn realAnswer expectedAnswer = do
    let len = A.length (A.flatten expectedAnswer)
        one = constant 1.0
        bp = A.zipWith (\a -> \b -> (b / a) - ((one - b) / (one - a))) realAnswer expectedAnswer
    A.map (\x -> - x / (A.fromIntegral len :: Exp Double)) bp
