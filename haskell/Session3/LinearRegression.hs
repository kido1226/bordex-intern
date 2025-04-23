{-# LANGUAGE BlockArguments #-}

module LinearRegression (runLinearRegression) where

import Control.Monad (zipWithM_)
import Torch.Functional as F
import Torch.Tensor (Tensor, asTensor, asValue)

ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])

xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

m :: Tensor
m = asTensor (15.0 :: Float)

rate :: Tensor -- learning rate
rate = asTensor (0.00001 :: Float)

epoch :: Int -- number of iterate
epoch = 100

linear ::
  -- | parameters ([a, b]: 1 × 2, c: scalar)
  (Tensor, Tensor) ->
  -- | data x: 1 × 10
  Tensor ->
  -- | z: 1 × 10
  Tensor
linear (slope, intercept) input = F.add (F.mul slope input) intercept

-- slope and input are scalar

cost ::
  -- | grand truth: 1 × 10
  Tensor ->
  -- | estimated values: 1 × 10
  Tensor ->
  -- | loss: scalar
  Tensor
cost z z' =
  let diffs = F.sub z' z
      squared = F.pow (2.0 :: Float) diffs
   in F.div (F.sumAll squared) (m * 2.0)

calculateNewA ::
  [Tensor] ->
  Tensor
calculateNewA [a, b] =
  let estimatedY = linear (a, b) xs
      diffs = F.sub estimatedY ys
      grad = F.div (F.sumAll (F.mul xs diffs)) m
   in F.sub a (F.mul grad rate)

calculateNewB ::
  [Tensor] ->
  Tensor
calculateNewB [a, b] =
  let estimatedY = linear (a, b) xs
      diffs = F.sub estimatedY ys
      grad = F.div (F.sumAll diffs) m
   in F.sub b (F.mul grad rate)

train :: Int -> Tensor -> Tensor -> IO (Tensor, Tensor)
train 0 a b = return (a, b)
train n a b = do
  let a' = calculateNewA [a, b]
  let b' = calculateNewB [a, b]
  let cost' = cost ys (linear (a', b') xs)
  putStrLn $ "Epoch " ++ show (epoch - n + 1) ++ ": cost = " ++ show (asValue cost' :: Float)
  train (n - 1) a' b'

-- if (asValue (cost ys (linear (a', b') xs)) :: Float) < 0.001
--    then (a', b')
--    else train (a', b')

-- main :: IO ()
runLinearRegression =
  do
    -- Below are pseudo code
    let sampleA = asTensor (1.0 :: Float)
    let sampleB = asTensor (1.0 :: Float)

    -- Iterate through the provided xs and ys data.
    trainedData <- train epoch sampleA sampleB
    print trainedData

    -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.

    let estimatedY = linear trainedData xs

    let ys' = (asValue ys :: [Float])
    let estimatedY' = (asValue estimatedY :: [Float])

    zipWithM_
      ( \a b -> do
          putStrLn $ "correct answer: " ++ show b
          putStrLn $ "estimated: " ++ show a
          putStrLn "******"
      )
      estimatedY'
      ys'

-- Expected outputs:
-- correct answer: 148
-- estimated: ?

-- correct answer: 186
-- ...