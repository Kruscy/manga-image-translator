$path = "manga_translator\translators\chatgpt.py"

$content = Get-Content $path -Raw

# remove temperature

$content = $content -replace "temperature=self.temperature,", ""

# replace max_tokens

$content = $content -replace "max_tokens=self.max_tokens", "max_completion_tokens=self.max_tokens"

# add fallback after message.content

$content = $content -replace "text = response.choices[0].message.content", @"
text = None

try:
text = response.choices[0].message.content
except:
pass

if (-not $text -and $response.output_text) {
$text = $response.output_text
}

if (-not $text) {
$text = $response | Out-String
}
"@

Set-Content $path $content

Write-Host "chatgpt.py patched for GPT-5-nano"
