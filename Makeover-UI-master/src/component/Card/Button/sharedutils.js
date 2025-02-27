export const getShadesAndEndpoint = (type) => {
  if (!["lipstick", "lip_liner", "eyeliner", "eyeshadow"].includes(type)) {
    throw new Error("Invalid parameter: type should be 'lipstick', 'lip_liner', 'eyeliner', or 'eyeshadow'.");
  }

  const matteShades = [
    { name: "M_01_700415", color: "#700415" },
  { name: "M_02_81071C", color: "#81071C" },
  { name: "M_03_8F131C", color: "#8F131C" },
  { name: "M_04_831E26", color: "#831E26" },
  { name: "M_05_66161A", color: "#66161A" },
  { name: "M_06_3C1213", color: "#3C1213" },
  { name: "M_07_401310", color: "#401310" },
  { name: "M_08_6E182E", color: "#6E182E" },
  { name: "M_09_801620", color: "#801620" },
  { name: "M_10_A95B57", color: "#A95B57" },
  { name: "M_11_5D0D10", color: "#5D0D10" },
  { name: "M_12_983445", color: "#983445" },
  { name: "M_13_60262F", color: "#60262F" },
  { name: "M_14_6E263D", color: "#6E263D" },
  { name: "M_15_812937", color: "#812937" },
  { name: "M_16_BF4951", color: "#BF4951" },
  { name: "M_17_5F2E23", color: "#5F2E23" },
  { name: "M_18_481F1B", color: "#481F1B" },
  { name: "M_19_7B221C", color: "#7B221C" },
  { name: "M_20_B85957", color: "#B85957" },
  { name: "M_21_8A3F3C", color: "#8A3F3C" },
  { name: "M_22_913B39", color: "#913B39" },
  { name: "M_23_6D2B1C", color: "#6D2B1C" },
  { name: "M_24_5D202A", color: "#5D202A" },
  ];

  const glossyShades = [...matteShades]; // Same shades for glossy

  // ✅ NEW: Lip Liner Shades
  const lipLinerShades = [
    { name: "L_01_5E1F1F", color: "#5E1F1F" }, // Dark Cocoa
    { name: "L_02_431616", color: "#431616" }, // Espresso Brown
    { name: "L_03_7B221C", color: "#7B221C" }, // Deep Wine
    { name: "L_04_6D2B1C", color: "#6D2B1C" }, // Mahogany
    { name: "L_05_913B39", color: "#913B39" }, // Rich Burgundy
  ];

  const eyelinerShades = [
    { name: "E_07_DarkPlum", color: "#4B0082" },
    { name: "E_08_DeepTeal", color: "#014B43" },
    { name: "E_09_DarkEmerald", color: "#046307" },
    { name: "E_10_Charcoal", color: "#36454F" },
    { name: "E_11_800020", color: "#800020" },
    { name: "E_12_SmokyQuartz", color: "#433E3F" },
    { name: "E_13_Mahogany", color: "#3A0200" },
    { name: "E_14_000080", color: "#000080" },
    { name: "E_15_DeepWine", color: "#522E3A" },
    { name: "E_16_DarkOlive", color: "#3B5323" },
  ];

  const eyeshadowShades = [
    { name: "E_01_8B3A62", color: "#8B3A62" },
    { name: "E_02_7A264D", color: "#7A264D" },
    { name: "E_03_5E1F3F", color: "#5E1F3F" },
    { name: "E_04_4A192E", color: "#4A192E" },
    { name: "E_05_983D3D", color: "#983D3D" },
    { name: "E_06_722929", color: "#722929" },
    { name: "E_07_5C1F1F", color: "#5C1F1F" },
    { name: "E_08_431616", color: "#431616" },
    { name: "E_09_8C4F2B", color: "#8C4F2B" },
    { name: "E_10_703B1E", color: "#703B1E" },
  ];

  switch (type) {
    case "lipstick":
      return { matteShades, glossyShades, endpoint: "lips" };
    case "lip_liner": // ✅ NEW CASE FOR LIP LINER
    return { shades: lipLinerShades, endpoint: "lip_liner" }; 
    case "eyeliner":
      return { shades: eyelinerShades, endpoint: "eyeliner" };
    case "eyeshadow":
      return { shades: eyeshadowShades, endpoint: "eyeshadow" };
    default:
      throw new Error("Unexpected type provided.");
  }
};
