{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Vect ( Vect(..), vaddv, mmulv, extract, takeV, dropV) where

import ML.ANN.Mat
import Data.Array.Accelerate as A
import Prelude as P

data Vect a where
    VectI :: Acc (Matrix Double) -> Vect InputSize -- We're using Matrices instead of vectors because it's going to be turned into a matrix later on anyways.
    VectO :: Acc (Matrix Double) -> Vect OutputSize

instance Show (Vect a) where
    show (VectI a) = show a
    show (VectO a) = show a

vaddv :: Vect a -> Vect a -> Vect a
vaddv (VectI a) (VectI b) = VectI (A.zipWith (+) a b)

mmulv :: Mat a b -> Vect b -> Vect a
mmulv (MatIO mat) (VectO vect) = do
    let sh = shape mat :: Exp (Z:.Int:.Int)
        (Z:._:.numOutputs) = unlift sh :: Z:.Exp Int:.Exp Int
        replicated = A.replicate (A.lift (Z:.numOutputs:.All)) (A.flatten vect) 
        transposed = A.transpose replicated
        retVect = A.sum (A.zipWith (+) mat transposed)
    VectI ((A.replicate (constant (Z:.All:.(1::Int))) retVect) :: Acc (Matrix Double))

mmulv (MatOI mat) (VectI vect) = do
    let sh = shape mat :: Exp (Z:.Int:.Int)
        (Z:.numInputs:._) = unlift sh :: Z:.Exp Int:.Exp Int
        replicated = A.replicate (A.lift (Z:.All:.numInputs)) (A.flatten vect) 
        transposed = A.transpose replicated
        retVect = A.sum (A.zipWith (+) mat transposed)
    VectO ((A.replicate (constant (Z:.All:.(1::Int))) retVect) :: Acc (Matrix Double))

extract :: Vect a -> Acc (Matrix Double)
extract (VectI a) = a
extract (VectO a) = a

takeV :: Vect a -> Exp Int -> Acc (Matrix Double)
takeV (VectI a) i = A.take i (A.transpose a)
takeV (VectO a) i = A.take i (A.transpose a)

dropV :: Vect a -> Exp Int -> Acc (Matrix Double)
dropV (VectI a) i = A.drop i a
dropV (VectO a) i = A.drop i a
