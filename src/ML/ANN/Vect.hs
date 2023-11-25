{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Vect ( Vect(..), vaddv, mmulv, extractVect, takeV, dropV, vxv, vmulv, vsubv, smulv, vzipw, vmap) where

import ML.ANN.Mat
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import Prelude as P
import System.IO

data Vect a where
    VectI :: Acc (Matrix Double) -> Vect InputSize -- We're using Matrices instead of vectors because it's going to be turned into a matrix later on anyways.
    VectO :: Acc (Matrix Double) -> Vect OutputSize

instance Show (Vect a) where
    show (VectI a) = show a
    show (VectO a) = show a

vaddv :: Vect a -> Vect a -> Vect a
vaddv (VectI a) (VectI b) = VectI (A.zipWith (+) a b)
vaddv (VectO a) (VectO b) = VectO (A.zipWith (+) a b)

vsubv :: Vect a -> Vect a -> Vect a
vsubv (VectI a) (VectI b) = VectI (A.zipWith (-) a b)
vsubv (VectO a) (VectO b) = VectO (A.zipWith (-) a b)

mmulv :: Mat a b -> Vect b -> Vect a
mmulv (MatIO mat) (VectO vect) = do
    let sh = shape mat :: Exp (Z:.Int:.Int)
        (Z:.numInputs:._) = unlift sh :: Z:.Exp Int:.Exp Int
        replicated = A.replicate (A.lift (Z:.numInputs:.All)) (A.flatten vect) 
        retVect = A.sum (A.zipWith (*) mat replicated)
    VectI ((A.replicate (constant (Z:.All:.(1::Int))) retVect) :: Acc (Matrix Double))

mmulv (MatOI mat) (VectI vect) = do
    let sh = shape mat :: Exp (Z:.Int:.Int)
        (Z:.numOutputs:._) = unlift sh :: Z:.Exp Int:.Exp Int
        replicated = A.replicate (A.lift (Z:.numOutputs:.All)) (A.flatten vect) 
        retVect = A.sum (A.zipWith (*) mat replicated)
    VectO ((A.replicate (constant (Z:.All:.(1::Int))) retVect) :: Acc (Matrix Double))

vxv :: Vect a -> Vect b -> Mat b a
vxv (VectO a) (VectI b) = do
    let ( Z :. aSize :. _) = A.unlift (A.shape a) :: Z:.Exp Int:.Exp Int
        ( Z :. bSize :. _) = A.unlift (A.shape b) :: Z:.Exp Int:.Exp Int
        aMat = A.replicate (A.lift (Z:.All:.bSize)) (A.flatten a)
        bMat = A.replicate (A.lift (Z:.aSize:.All)) (A.flatten b)
        retMat = A.zipWith (*) aMat bMat
    MatIO (A.transpose retMat)
vxv (VectI a) (VectO b) = do
    let ( Z :. aSize :. _) = A.unlift (A.shape a) :: Z:.Exp Int:.Exp Int
        ( Z :. bSize :. _) = A.unlift (A.shape b) :: Z:.Exp Int:.Exp Int
        aMat = A.replicate (A.lift (Z:.All:.bSize)) (A.flatten a)
        bMat = A.replicate (A.lift (Z:.aSize:.All)) (A.flatten b)
        retMat = A.zipWith (*) aMat bMat
    MatOI (A.transpose retMat)

vmulv :: Vect a -> Vect a -> Vect a
vmulv (VectI a) (VectI b) = VectI (A.zipWith (*) a b)
vmulv (VectO a) (VectO b) = VectO (A.zipWith (*) a b)

smulv :: Exp Double -> Vect a -> Vect a
smulv s (VectI a) = VectI (A.map (\x -> x * s) a)
smulv s (VectO a) = VectO (A.map (\x -> x * s) a)

extractVect :: Vect a -> Acc (Matrix Double)
extractVect (VectI a) = a
extractVect (VectO a) = a

takeV :: Vect a -> Exp Int -> Acc (Matrix Double)
takeV (VectI a) i = A.take i (A.transpose a)
takeV (VectO a) i = A.take i (A.transpose a)

dropV :: Vect a -> Exp Int -> Acc (Matrix Double)
dropV (VectI a) i = A.drop i (A.transpose a)
dropV (VectO a) i = A.drop i (A.transpose a)


vzipw :: (Exp Double -> Exp Double -> Exp Double) -> Vect a -> Vect a -> Vect a
vzipw fn (VectI a) (VectI b) = VectI (A.zipWith fn a b)
vzipw fn (VectO a) (VectO b) = VectO (A.zipWith fn a b)

vmap :: (Exp Double -> Exp Double) -> Vect a -> Vect a
vmap fn (VectO v) = VectO (A.map fn v)
vmap fn (VectI v) = VectI (A.map fn v)
