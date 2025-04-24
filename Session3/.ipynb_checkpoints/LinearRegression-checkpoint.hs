{-# LANGUAGE BlockArguments #-}

module LinearRegression where

import Torch.Functional (add, matmul, mul, transpose2D)
import Torch.Tensor (Tensor, asTensor)

ys :: Tensor
ys = asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])

xs :: Tensor
xs = asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear ::
  -- | parameters ([a, b]: 1 × 2, c: scalar)
  (Tensor, Tensor) ->
  -- | data x: 1 × 10
  Tensor ->
  -- | z: 1 × 10
  Tensor
linear (slope, intercept) input = add (mul slope input) intercept

-- slope and input are scalar

main :: IO ()
main =
  do
    -- Below are pseudo code
    let sampleA = asTensor (0.555 :: Float)
    let sampleB = asTensor (94.585026 :: Float)

    -- Iterate through the provided xs and ys data.
    -- For each pair, convert x to a tensor, calculate the estimatedY using your linear function with the provided sampleA and sampleB, and print both the correct y and the estimatedY.

    let estimatedY = linear (sampleA, sampleB) xs

    print ys
    print estimatedY

-- forM ys $ \a -> do
-- 	forM estimatedY $ \b -> do
-- 		print "correct answer:" + show a
-- 		print "estimated: " + show b
-- 		print "******"

-- Expected outputs:
-- correct answer: 148
-- estimated: ?

-- correct answer: 186
-- ...