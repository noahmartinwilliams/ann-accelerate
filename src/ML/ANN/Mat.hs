{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Mat ( Mat(..), InputSize, OutputSize, maddm, transp ) where

import Data.Array.Accelerate as A
import Prelude as P

data InputSize
data OutputSize

data Mat a b where
    MatIO :: Acc (Matrix Double) -> Mat InputSize OutputSize
    MatOI :: Acc (Matrix Double) -> Mat OutputSize InputSize

instance Show (Mat a b) where
    show (MatIO a) = show a
    show (MatOI a) = show a

maddm :: Mat a b -> Mat a b -> Mat a b
maddm (MatIO a) (MatIO b) = MatIO (A.zipWith (\x -> \y -> x + y) a b)
maddm (MatOI a) (MatOI b) = MatOI (A.zipWith (\x -> \y -> x + y) a b)

transp :: Mat a b -> Mat b a
transp (MatIO a) = MatOI (A.transpose a)
transp (MatOI a) = MatIO (A.transpose a)
