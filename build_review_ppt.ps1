param(
  [string]$OutputPath = (Join-Path $PSScriptRoot "Investor_Inventor_Review_Deck.pptx")
)

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$assetScript = Join-Path $root "generate_ppt_assets.py"
$assetJson = & python $assetScript
$data = $assetJson | ConvertFrom-Json
$assets = $data.assets
$comparison = $data.comparison
$best = $comparison[0]

function Get-RgbInt {
  param([int]$R, [int]$G, [int]$B)
  return $R + ($G * 256) + ($B * 65536)
}

$ColorBg = Get-RgbInt 8 19 29
$ColorPanel = Get-RgbInt 16 36 53
$ColorPanel2 = Get-RgbInt 20 42 62
$ColorLine = Get-RgbInt 35 64 90
$ColorText = Get-RgbInt 238 247 255
$ColorMuted = Get-RgbInt 157 181 201
$ColorPrimary = Get-RgbInt 55 213 255
$ColorSecondary = Get-RgbInt 255 177 74

function New-BlankSlide {
  param($Presentation)
  $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 12)
  $slide.FollowMasterBackground = $false
  $slide.Background.Fill.Solid()
  $slide.Background.Fill.ForeColor.RGB = $ColorBg
  $accent = $slide.Shapes.AddShape(1, 0, 0, 960, 6)
  $accent.Fill.Solid()
  $accent.Fill.ForeColor.RGB = $ColorPrimary
  $accent.Line.Visible = 0
  return $slide
}

function Add-TextBox {
  param(
    [object]$Slide,
    [double]$Left,
    [double]$Top,
    [double]$Width,
    [double]$Height,
    [string]$Text,
    [int]$FontSize = 20,
    [int]$Color = $ColorText,
    [bool]$Bold = $false,
    [int]$Alignment = 1
  )
  $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
  $shape.TextFrame.WordWrap = $true
  $shape.TextFrame.MarginLeft = 4
  $shape.TextFrame.MarginRight = 4
  $shape.TextFrame.MarginTop = 2
  $shape.TextFrame.MarginBottom = 2
  $shape.TextFrame.TextRange.Text = $Text
  $shape.TextFrame.TextRange.Font.Name = "Segoe UI"
  $shape.TextFrame.TextRange.Font.Size = $FontSize
  $shape.TextFrame.TextRange.Font.Bold = $Bold
  $shape.TextFrame.TextRange.Font.Color.RGB = $Color
  $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
  return $shape
}

function Add-Box {
  param(
    [object]$Slide,
    [double]$Left,
    [double]$Top,
    [double]$Width,
    [double]$Height,
    [string]$Text,
    [int]$Fill = $ColorPanel,
    [int]$Line = $ColorLine,
    [int]$FontSize = 18,
    [int]$TextColor = $ColorText,
    [bool]$Bold = $false,
    [int]$Alignment = 1
  )
  $shape = $Slide.Shapes.AddShape(1, $Left, $Top, $Width, $Height)
  $shape.Fill.Solid()
  $shape.Fill.ForeColor.RGB = $Fill
  $shape.Line.ForeColor.RGB = $Line
  $shape.Line.Weight = 1.5
  $shape.TextFrame.WordWrap = $true
  $shape.TextFrame.MarginLeft = 10
  $shape.TextFrame.MarginRight = 10
  $shape.TextFrame.MarginTop = 10
  $shape.TextFrame.MarginBottom = 10
  $shape.TextFrame.TextRange.Text = $Text
  $shape.TextFrame.TextRange.Font.Name = "Segoe UI"
  $shape.TextFrame.TextRange.Font.Size = $FontSize
  $shape.TextFrame.TextRange.Font.Bold = $Bold
  $shape.TextFrame.TextRange.Font.Color.RGB = $TextColor
  $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Alignment
  return $shape
}

function Add-Image {
  param(
    [object]$Slide,
    [string]$Path,
    [double]$Left,
    [double]$Top,
    [double]$Width,
    [double]$Height
  )
  if (-not (Test-Path $Path)) {
    throw "Missing image: $Path"
  }
  $Slide.Shapes.AddPicture($Path, 0, -1, $Left, $Top, $Width, $Height) | Out-Null
}

function Add-Title {
  param(
    [object]$Slide,
    [string]$Title,
    [string]$Subtitle = ""
  )
  Add-TextBox $Slide 40 36 860 58 $Title 22 $ColorPrimary $true 1 | Out-Null
  Add-TextBox $Slide 40 82 880 74 $Subtitle 32 $ColorText $true 1 | Out-Null
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$ppt.DisplayAlerts = 1
$presentation = $ppt.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

# Slide 1
$slide = New-BlankSlide $presentation
Add-TextBox $slide 42 82 610 82 "Investor-Inventor Matchmaking System" 30 $ColorText $true 1 | Out-Null
Add-TextBox $slide 42 176 600 110 "Machine learning based recommendation platform for startup funding and discovery" 18 $ColorMuted $false 1 | Out-Null
Add-Box $slide 42 308 600 108 "Review focus: model training, evaluation metrics, ROC/PR graphs, confusion matrices, and final model selection." $ColorPanel2 $ColorLine 18 $ColorText $false 1 | Out-Null
Add-Box $slide 690 112 230 175 ("Best model`n`n" + $best.name + "`nDecision score: " + ('{0:N2}%' -f ($best.decision_score * 100))) $ColorPanel $ColorPrimary 20 $ColorText $true 1 | Out-Null
Add-TextBox $slide 42 450 640 34 "Dashboard-backed demo deck prepared for 2nd review." 16 $ColorMuted $false 1 | Out-Null

# Slide 2
$slide = New-BlankSlide $presentation
Add-Title $slide "Problem and Objective" "Why this system is needed"
$body = @"
• Investors need a faster way to discover relevant startup ideas.
• Inventors need a better way to reach suitable investors.
• Manual matching is slow, subjective, and difficult to scale.
• The objective is to recommend strong investor-inventor pairs using machine learning.
• The platform also supports chat, likes/dislikes, and profile review.
"@
Add-TextBox $slide 56 172 560 260 $body 20 $ColorText $false 1 | Out-Null
Add-Box $slide 650 176 250 248 "Why it matters`n`nA good recommender reduces noise, saves time, and increases the chance of a meaningful funding connection." $ColorPanel2 $ColorLine 19 $ColorText $true 1 | Out-Null

# Slide 3
$slide = New-BlankSlide $presentation
Add-Title $slide "Dataset and Feature Engineering" "What the model learns from"
$body = @"
• investor.csv provides funds, domain interest, location, and risk appetite.
• inventor.csv provides startup domain, technology, location, risk, and funding need.
• history.csv provides previous interaction outcomes.
• Feature engineering adds domain match, location match, risk gap, affordability ratio, and text similarity.
• Historical match statistics help the model learn which combinations worked before.
"@
Add-TextBox $slide 56 168 560 320 $body 19 $ColorText $false 1 | Out-Null
Add-Box $slide 650 172 250 260 "Feature importance`n`nThese engineered signals are more useful than raw IDs alone because they describe how well the two profiles fit each other." $ColorPanel2 $ColorLine 19 $ColorText $true 1 | Out-Null

# Slide 4
$slide = New-BlankSlide $presentation
Add-Image $slide $assets.train_test_split 20 20 920 500
Add-TextBox $slide 40 500 860 26 "Train/Test split diagram: training data teaches the model, test data checks unseen performance." 16 $ColorMuted $false 1 | Out-Null

# Slide 5
$slide = New-BlankSlide $presentation
Add-Title $slide "Models Compared" "Multiple algorithms were trained on the same split"
$body = @"
• Logistic Regression: simple baseline and easy to interpret.
• Random Forest: ensemble model that handles non-linear patterns well.
• Gradient Boosting: sequential ensemble that often gives strong ranking performance.
• Decision Tree: interpretable but usually less stable than ensembles.
• Comparing all models helps us choose the most reliable recommender.
"@
Add-TextBox $slide 56 174 560 300 $body 19 $ColorText $false 1 | Out-Null
Add-Box $slide 650 176 250 250 "Why compare models?`n`nBecause a recommendation system should be evaluated on more than one metric. Different models can trade accuracy, precision, recall, and curve quality differently." $ColorPanel2 $ColorLine 18 $ColorText $true 1 | Out-Null

# Slide 6
$slide = New-BlankSlide $presentation
Add-Image $slide $assets.performance_table 14 18 932 510

# Slide 7
$slide = New-BlankSlide $presentation
Add-Image $slide $assets.confusion_matrices 14 18 932 510

# Slide 8
$slide = New-BlankSlide $presentation
Add-Image $slide $assets.roc_curve 14 18 932 510

# Slide 9
$slide = New-BlankSlide $presentation
Add-Image $slide $assets.pr_curve 14 18 932 510

# Slide 10
$slide = New-BlankSlide $presentation
Add-Title $slide "How the Best Model Is Chosen" "Model selection logic"
$formula = @"
Decision Score = (Accuracy + Precision + Recall + F1 + ROC AUC + PR AUC) / 6
"@
Add-Box $slide 58 176 840 74 $formula $ColorPanel2 $ColorPrimary 24 $ColorText $true 1 | Out-Null
$explain = @"
• The best model is not chosen by accuracy alone.
• The composite decision score gives equal importance to all six metrics.
• ROC AUC and PR AUC are important because they show ranking quality across thresholds.
• If two models are very close, the curve metrics break the tie.
• In our run, Gradient Boosting edges out the others by the final decision score.
"@
Add-TextBox $slide 58 280 520 220 $explain 19 $ColorText $false 1 | Out-Null
Add-Box $slide 620 280 260 178 ("Final ranking`n`n1. " + $comparison[0].name + "  " + ('{0:N2}%' -f ($comparison[0].decision_score * 100)) + "`n2. " + $comparison[1].name + "  " + ('{0:N2}%' -f ($comparison[1].decision_score * 100)) + "`n3. " + $comparison[2].name + "  " + ('{0:N2}%' -f ($comparison[2].decision_score * 100)) + "`n4. " + $comparison[3].name + "  " + ('{0:N2}%' -f ($comparison[3].decision_score * 100))) $ColorPanel $ColorLine 16 $ColorText $true 1 | Out-Null

# Slide 11
$slide = New-BlankSlide $presentation
Add-Title $slide "Final Result and Demo Screenshots" "What the platform delivers"
$body = @"
• The system ranks investor-inventor pairs using machine learning.
• Gradient Boosting is selected as the best model under the balanced decision rule.
• The platform includes login/signup, dashboards, chat, likes/dislikes, and an equity calculator.
• Insert a login or dashboard screenshot in the boxes below if you want a full demo slide.
"@
Add-TextBox $slide 46 170 520 240 $body 18 $ColorText $false 1 | Out-Null
Add-Box $slide 610 164 300 140 "Insert investor dashboard screenshot here" $ColorPanel2 $ColorLine 18 $ColorMuted $true 2 | Out-Null
Add-Box $slide 610 330 300 140 "Insert analytics dashboard screenshot here" $ColorPanel2 $ColorLine 18 $ColorMuted $true 2 | Out-Null

if (Test-Path $OutputPath) {
  Remove-Item $OutputPath -Force
}

$presentation.SaveAs($OutputPath, 24)
$presentation.Close()
$ppt.Quit()

Write-Host "PPT created: $OutputPath"
