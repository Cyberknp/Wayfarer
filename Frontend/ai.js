function getRiskLevel(zone) {
  const hour = new Date().getHours();

  let score = 0;

  // Time based logic (night = risky)
  if (hour >= 20 || hour <= 5) score += 2;

  // Zone base weight
  if (zone.level === "red") score += 3;
  if (zone.level === "orange") score += 2;
  if (zone.level === "green") score += 1;

  // Fake anomaly detection
  if (Math.random() > 0.7) score += 2;

  if (score >= 5) return "HIGH";
  if (score >= 3) return "MEDIUM";
  return "LOW";
}
